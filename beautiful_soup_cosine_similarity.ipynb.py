
# coding: utf-8

# In[1]:


from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


# # Question 
# #### Step 1) Read the following articles using Beautifulsoup Python library provided to you
# #### Step 2) Using the above code build the Cosine Similarity Matrix using CountVectorizer and TFIDFVectorizer 
# #### Step 3) Plot Similarity Matrix using CountVectorizer and TFIDFVectorizer 
# 
# ##### URLs:
# <li> https://en.wikipedia.org/wiki/World_War_II
# <li> https://en.wikipedia.org/wiki/World_War_I 
# <li> https://en.wikipedia.org/wiki/War_of_1812
# <li> https://en.wikipedia.org/wiki/Basketball
# <li> https://en.wikipedia.org/wiki/Association_football

# In[34]:


import nltk
import urllib.parse
import urllib.request
#from urllib.request import Request, urlopen
from urllib.error import URLError
from bs4 import BeautifulSoup
#nltk.download('punkt')


# In[35]:


urlList = ['https://en.wikipedia.org/wiki/World_War_II',
         'https://en.wikipedia.org/wiki/World_War_I',
         'https://en.wikipedia.org/wiki/War_of_1812',
         'https://en.wikipedia.org/wiki/Basketball',
         'https://en.wikipedia.org/wiki/Association_football']


# In[36]:


# Table mapping response codes to messages; entries have the
# form {code: (shortmessage, longmessage)}.
responses = {
    100: ('Continue', 'Request received, please continue'),
    101: ('Switching Protocols',
          'Switching to new protocol; obey Upgrade header'),

    200: ('OK', 'Request fulfilled, document follows'),
    201: ('Created', 'Document created, URL follows'),
    202: ('Accepted',
          'Request accepted, processing continues off-line'),
    203: ('Non-Authoritative Information', 'Request fulfilled from cache'),
    204: ('No Content', 'Request fulfilled, nothing follows'),
    205: ('Reset Content', 'Clear input form for further input.'),
    206: ('Partial Content', 'Partial content follows.'),

    300: ('Multiple Choices',
          'Object has several resources -- see URI list'),
    301: ('Moved Permanently', 'Object moved permanently -- see URI list'),
    302: ('Found', 'Object moved temporarily -- see URI list'),
    303: ('See Other', 'Object moved -- see Method and URL list'),
    304: ('Not Modified',
          'Document has not changed since given time'),
    305: ('Use Proxy',
          'You must use proxy specified in Location to access this '
          'resource.'),
    307: ('Temporary Redirect',
          'Object moved temporarily -- see URI list'),

    400: ('Bad Request',
          'Bad request syntax or unsupported method'),
    401: ('Unauthorized',
          'No permission -- see authorization schemes'),
    402: ('Payment Required',
          'No payment -- see charging schemes'),
    403: ('Forbidden',
          'Request forbidden -- authorization will not help'),
    404: ('Not Found', 'Nothing matches the given URI'),
    405: ('Method Not Allowed',
          'Specified method is invalid for this server.'),
    406: ('Not Acceptable', 'URI not available in preferred format.'),
    407: ('Proxy Authentication Required', 'You must authenticate with '
          'this proxy before proceeding.'),
    408: ('Request Timeout', 'Request timed out; try again later.'),
    409: ('Conflict', 'Request conflict.'),
    410: ('Gone',
          'URI no longer exists and has been permanently removed.'),
    411: ('Length Required', 'Client must specify Content-Length.'),
    412: ('Precondition Failed', 'Precondition in headers is false.'),
    413: ('Request Entity Too Large', 'Entity is too large.'),
    414: ('Request-URI Too Long', 'URI is too long.'),
    415: ('Unsupported Media Type', 'Entity body in unsupported format.'),
    416: ('Requested Range Not Satisfiable',
          'Cannot satisfy request range.'),
    417: ('Expectation Failed',
          'Expect condition could not be satisfied.'),

    500: ('Internal Server Error', 'Server got itself in trouble'),
    501: ('Not Implemented',
          'Server does not support this operation'),
    502: ('Bad Gateway', 'Invalid responses from another server/proxy.'),
    503: ('Service Unavailable',
          'The server cannot process the request due to a high load'),
    504: ('Gateway Timeout',
          'The gateway server did not receive a timely response'),
    505: ('HTTP Version Not Supported', 'Cannot fulfill request.'),
    }


# In[49]:


def gettokens (responses,url):
    #url = "https://en.wikipedia.org/wiki/World_War_II"
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    values = {'name': 'Michael Foord',
              'location': 'Northampton',
              'language': 'Python' }
    headers = {'User-Agent': user_agent}
    data = urllib.parse.urlencode(values)
    data = data.encode('ascii')
    
    req = urllib.request.Request(url, data, headers)
    try:
        #urllib.request.urlopen(req)
        with urllib.request.urlopen(req) as response:
            html = response.read()
    except URLError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', responses[e.code])
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
    else:
        print("everything is fine")
        web_str = BeautifulSoup(html, "lxml").get_text()
        #web_tokens = nltk.word_tokenize(web_str)
    
    return web_str


# In[50]:


url_0 = gettokens(responses,urlList[0])
url_1 = gettokens(responses,urlList[1])
url_2 = gettokens(responses,urlList[2])
url_3 = gettokens(responses,urlList[3])
url_4 = gettokens(responses,urlList[4])


# In[53]:


docs = [url_0,url_1,url_2,url_3,url_4]


# In[56]:


vect = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=20)
X = vect.fit_transform(docs)

count_df = DataFrame(X.A, columns=vect.get_feature_names())
#print (count_df)

count_ary = count_df.values
#print (count_ary)

#Count Vectorizer
correlation= 1-pairwise_distances(count_ary, metric='cosine')
print (correlation)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=0, vmax=1)
fig.colorbar(cax)
plt.show()


# In[57]:


#TFIDF Vectorizer
f = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, stop_words="english", analyzer='word', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=20)
Xi = f.fit_transform(docs)

tfid_df = DataFrame(Xi.A, columns=f.get_feature_names())
#print (tfid_df)

tfid_ary = tfid_df.values
#print (tfid_ary)

correlation= 1-pairwise_distances(tfid_ary, metric='cosine')
print (correlation)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=0, vmax=1)
fig.colorbar(cax)
plt.show()

