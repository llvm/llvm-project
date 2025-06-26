@echo off

:: This script is used to install the .NET SDK.
:: It will also invoke the SDK with any provided arguments.

powershell -ExecutionPolicy ByPass -NoProfile -command "& """%~dp0dotnet.ps1""" %*"
exit /b %ErrorLevel%
