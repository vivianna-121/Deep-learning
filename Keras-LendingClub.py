#!/usr/bin/env python
# coding: utf-8

# # LendingClub Project 
# 
# ## Background
# 
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ###  Goal
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model that can predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. 
# 
# The "loan_status" column contains the label.
# 
# ### Data Overview

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----

# ## Starter Code
# 
# #### Create a function to look up feature information

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


print(data_info.loc['revol_util']['Description'])


# In[4]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[5]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[7]:


df.info()


# 
# 
# # Section 1: Exploratory Data Analysis
# 
# **OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data**
# 
# 
# ----

# **TASK: Since we will be attempting to predict loan_status, create a countplot as shown below.**

# In[11]:


sns.countplot(df['loan_status'])


# The datasset is inbalanced.

# **TASK: Create a histogram of the loan_amnt column.**

# In[22]:


#plt.hist(df['loan_amnt'],bins=30)
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],bins=30,kde=False)


# **TASK: Let's explore correlation between the continuous feature variables. Calculate the correlation between all continuous numeric variables using .corr() method.**

# In[31]:


df.head()


# In[8]:


cor = df.corr()
cor


# **TASK: Visualize this using a heatmap.**    
# 

# In[9]:


plt.figure(figsize=(12,7))
ax = sns.heatmap(cor,annot=True)
#fix the size problem
plt.ylim(10, 0)

#OR:
#bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)


# **TASK: You should have noticed almost perfect correlation with the "installment" feature. Explore this feature further. Print out their descriptions and perform a scatterplot between them.**

# In[32]:


feat_info('installment')


# In[34]:


feat_info('loan_amnt')


# In[35]:


sns.scatterplot(x='installment',y='loan_amnt',data=df)


# This indicates there must be some formular to get the installment based on loan amount, which does quite make sense. 

# **TASK: Create a boxplot showing the relationship between the loan_status and the Loan Amount.**

# In[37]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# The box plots seem quite similar, further check the statistics manually in the next step.

# **TASK: Calculate the summary statistics for the loan amount, grouped by the loan_status.**

# In[10]:


df.groupby('loan_status')['loan_amnt'].describe()


# **TASK: Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?**

# In[64]:


#df.info()


# In[62]:


sorted(df.grade.unique())


# In[63]:


sorted(df.sub_grade.unique())


# **TASK: Create a countplot per grade. Set the hue to the loan_status label.**

# In[65]:


sns.countplot(x='grade',data=df,hue='loan_status')


# **TASK: Display a count plot per subgrade.**

# In[6]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df.sub_grade.unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm')


# **Create a similar plot, but set hue="loan_status".**

# In[7]:


plt.figure(figsize=(12,4))
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',hue='loan_status')


# **TASK: It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.**

# In[8]:


FG = df[(df['grade']=='F') | (df['grade']=='G')]
plt.figure(figsize=(12,4))
order = sorted(FG['sub_grade'].unique())
sns.countplot(x='sub_grade',data=FG,hue='loan_status',order=order)


# In[10]:


#df['loan_status']


# **TASK: Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[8]:


def repay(x):
    if x =='Fully Paid':
        return 1
    else:
        return 0

df['loan_repaid'] = df['loan_status'].apply(repay)
#df['loan_repaid']


# In[9]:


# OR: use the map function
#df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[19]:


df.head()


# **Create a bar plot showing the correlation of the numeric features to the new loan_repaid column.**

# In[8]:


df.corr()['loan_repaid'].sort_values()


# In[9]:


df.corr()['loan_amnt'].sort_values()


# In[24]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# ---
# ---
# # Section 2: Data PreProcessing
# 
# **Section Goals: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.**
# 
# 

# In[29]:


df.head()


# # Missing Data
# 
# **Let's explore this missing data columns. We use a variety of factors to decide whether or not they would be useful, to see if we should keep, discard, or fill in the missing data.**

# **TASK: What is the length of the dataframe?**

# In[30]:


len(df)


# **TASK: Create a Series that displays the total count of missing values per column.**

# In[32]:


df.isna().sum()


# **TASK: Convert this Series to be in term of percentage of the total DataFrame**

# In[10]:


100*df.isnull().sum()/len(df)


# **TASK: Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info() function from the top of this notebook.**

# In[40]:


feat_info('emp_title')


# In[41]:


feat_info('emp_length')


# **TASK: How many unique employment job titles are there?**

# In[11]:


len(df['emp_title'].unique())
#OR:
# df['emp_title'].nunique()


# In[12]:


df['emp_title'].unique()


# In[62]:


#df['emp_title'].value_counts()


# **TASK: Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.**

# In[10]:


df.drop('emp_title',axis=1,inplace=True)


# **TASK: Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.**

# In[11]:


df['emp_length'].unique()


# In[12]:


sorted(df['emp_length'].dropna().unique())


# In[16]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[17]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order)


# **TASK: Plot out the countplot with a hue separating Fully Paid vs Charged Off**

# In[77]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# **This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category didn't pay back their loan.**

# In[13]:


emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']


# In[14]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[16]:


emp_len = emp_co/(emp_fp+emp_co)
emp_len


# In[17]:


emp_len.plot(kind='bar')


# **TASK: Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.**

# In[18]:


df.drop('emp_length',axis=1,inplace=True)


# **TASK: Revisit the DataFrame to see what feature columns still have missing data.**

# In[19]:


df.isna().sum()


# **TASK: Review the title column vs the purpose column. Is this repeated information?**

# In[116]:


df['title'].head()


# In[117]:


df['purpose'].head()


# **TASK: The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.**

# In[20]:


df.drop('title',axis=1,inplace=True)


# 
# **TASK: Find out what the mort_acc feature represents**

# In[119]:


feat_info('mort_acc')


# In[122]:


df['mort_acc'].head()


# **TASK: Create a value_counts of the mort_acc column.**

# In[120]:


df['mort_acc'].value_counts()


# **TASK: There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in, such as a linear model, we could just fill it in based on the mean of the other columns, or we could even bin the columns into categories and then set NaN as its own category. There is no 100% correct approach! Let's review the other columsn to see which most highly correlates to mort_acc**

# In[21]:


print("Correlation with the mort_acc column")

df.corr()['mort_acc'].sort_values()


# **TASK: Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry.**

# In[22]:


#fill in null values with mean of the most correlated column
print('Mean of mort_acc column per total_acc')
df.groupby('total_acc').mean()['mort_acc']


# **Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above.**

# In[23]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[24]:


total_acc_avg[2.0]


# In[25]:


def fill (mort_acc,total_acc):
  
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[26]:


df['mort_acc'] = df.apply(lambda x: fill(x['mort_acc'],x['total_acc'] ), axis=1)


# In[27]:


df.isna().sum()


# **TASK: revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Go ahead and remove the rows that are missing those values in those columns with dropna().**

# In[28]:


#remove the rows that are missing those values in those columns 
df = df.dropna()


# In[29]:


df.isna().sum()


# ## Categorical Variables and Dummy Variables
# 
# 
# **TASK: List all the columns that are currently non-numeric.**
# 

# In[44]:


list(df.select_dtypes(exclude=np.number).columns)


# In[11]:


#OR:
list(df.select_dtypes(object).columns)


# ---
# **Let's now go through all the string features to see what we should do with them.**
# 
# ---
# 
# 
# ### term feature
# 
# **TASK: Convert the term feature into either a 36 or 60 integer numeric data type.**

# In[30]:


df['term'].head()


# In[31]:


df['term'][0].split()[0]


# In[32]:


#df['term']
df['term'] = df['term'].apply(lambda x: x.split()[0])


# In[33]:


df['term']=pd.to_numeric(df['term'])


# In[34]:


df['term'].head()


# In[13]:


#OR: use "int()"
#df['term'].apply(lambda term: int(term[:3]))


# ### grade feature
# 
# **TASK: We already know grade is part of sub_grade, so just drop the grade feature.**

# In[35]:


df.drop('grade',axis=1,inplace=True)


# **TASK: Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe.**

# In[36]:


df['sub_grade'].head()


# In[37]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[38]:


subgrade_dummies.head()


# In[69]:


df.columns


# In[39]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[34]:


df.columns


# ### verification_status, application_type,initial_list_status,purpose 
# **TASK: Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[40]:


dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)


# In[41]:


df = df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1)


# In[42]:


df = pd.concat([df,dummies],axis=1)


# ### home_ownership
# **TASK:Review the value_counts for the home_ownership column.**

# In[38]:


df['home_ownership'].value_counts()


# **TASK: Convert these to dummy variables, but [replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER.**

# In[43]:


df['home_ownership'] = df['home_ownership'].replace(('NONE','ANY'),'OTHER')


# In[44]:


df['home_ownership'].value_counts()


# In[45]:


dummies = pd.get_dummies(df['home_ownership'],drop_first=True)


# In[46]:


df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# ### address
# **TASK: Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.**

# In[47]:


df['address'][0].split()[-1]


# In[48]:


df['zip_code'] = df['address'].apply(lambda x: x.split()[-1])


# In[49]:


df['zip_code'].head()


# **TASK: Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the original zip_code column along with dropping the address column.**

# In[50]:


zip_dummy = pd.get_dummies(df['zip_code'],drop_first=True)


# In[51]:


df = pd.concat([df,zip_dummy],axis=1)


# In[52]:


df = df.drop(['zip_code','address'],axis=1)


# In[97]:


df.columns


# ### issue_d 
# 
# **TASK: This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**

# In[53]:


df = df.drop('issue_d',axis=1)


# ### earliest_cr_line
# **TASK: This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.**

# In[54]:


df['earliest_cr_line'][0][4:]


# In[55]:


df.select_dtypes(['object']).columns


# In[56]:


df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: x[4:])


# In[57]:


df['earliest_cr_line']=pd.to_numeric(df['earliest_cr_line'])
df['earliest_cr_line'].head()


# In[58]:


df.select_dtypes(['object']).columns


# ## Train Test Split

# **TASK: Import train_test_split from sklearn.**

# In[59]:


from sklearn.model_selection import train_test_split


# **TASK: drop the loan_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.**

# In[60]:


df = df.drop('loan_status',axis=1)


# In[61]:


df.columns


# **TASK: Set X and y variables to the .values of the features and label.**

# In[61]:


X=df.drop('loan_repaid',axis=1).values
y=df['loan_repaid'].values


# In[63]:


print(len(df))


# **TASK: Perform a train/test split with test_size=0.2 and a random_state of 101.**

# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ## Normalizing the Data
# 
# **TASK: Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data.**

# In[63]:


from sklearn.preprocessing import MinMaxScaler


# In[64]:


scaler = MinMaxScaler()


# In[65]:


X_train = scaler.fit_transform(X_train)


# In[66]:


X_test = scaler.transform(X_test)


# # Creating the Model
# 

# In[67]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[68]:


model = Sequential()

#input layer
model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))

#hidden layer
model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))

#output layer
model.add(Dense(1,activation='sigmoid'))

#compile nmodel
model.compile(loss='binary_crossentropy', optimizer='adam')


# **TASK: Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256.**

# In[69]:


model.fit(x=X_train,y=y_train,epochs=25,validation_data=(X_test,y_test),batch_size=256)


# **TASK: OPTIONAL: Save your model.**

# In[70]:


from tensorflow.keras.models import load_model


# In[71]:


model.save('full_data_project_model.h5')  


# # Section 3: Evaluating Model Performance.
# 
# **TASK: Plot out the validation loss versus the training loss.**

# In[72]:


loss = pd.DataFrame(model.history.history)
loss


# In[73]:


loss.plot()


# **TASK: Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# In[74]:


prediction = model.predict_classes(X_test)


# In[75]:


from sklearn.metrics import classification_report,confusion_matrix


# In[76]:


print("classification report")
print('\n')
print(classification_report(y_test,prediction))
print('\n')
print("confusion matrix")
print('\n')
print(confusion_matrix(y_test,prediction))


# **TASK: Predict for a new customer below, would you offer this person a loan?**

# In[77]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[82]:


new_customer = scaler.transform(new_customer.values.reshape(1,78))
predict = model.predict_classes(new_customer)
predict


# **TASK: Now check, did this person actually end up paying back their loan?**

# In[83]:


df.iloc[random_ind]['loan_repaid']

