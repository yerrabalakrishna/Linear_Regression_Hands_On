# Linear_Regression_Hands_On
Pridicts the values 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

###Read the csv file

data=pd.read_csv("annual-enterprise-survey-2024-financial-year-provisional.csv")
print(data.head())
print(data.describe())
print(data.shape)
print(data.tail(10))
print(data.info())

###Depandeis
one=data[['bedrooms','floors']]
print(one)

###Create a model
x=data[['bedrooms']]
y=data[['floors']]
data.plot(x='bedrooms',y='floors', color='pink')
plt.xlabel('bedrooms')
plt.ylabel('floors')
plt.show()

###Train test
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("-"*20)
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
###Create a model
model=LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)

###Prediction
y_pred=model.predict(X_test)
y_pred=pd.DataFrame(y_pred,columns=['Predicted'])
print(y_pred.size)
print(y_pred)
##Actual Value
print(y_test)

###Calculating the errors
print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
