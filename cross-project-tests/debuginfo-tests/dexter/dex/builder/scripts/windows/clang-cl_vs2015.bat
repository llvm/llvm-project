@echo OFF
setlocal EnableDelayedExpansion

call "%VS140COMNTOOLS%..\..\VC\bin\amd64\vcvars64.bat"

@echo OFF
setlocal EnableDelayedExpansion

for %%I in (%SOURCE_INDEXES%) do (
  %PATHTOCLANGCL% /c !COMPILER_OPTIONS_%%I! !SOURCE_FILE_%%I! /Fo!OBJECT_FILE_%%I!
  if errorlevel 1 goto :FAIL
)

%PATHTOCLANGCL% %LINKER_OPTIONS% %OBJECT_FILES% /Fe%EXECUTABLE_FILE%
if errorlevel 1 goto :FAIL
goto :END

:FAIL
echo FAILED
exit /B 1

:END
exit /B 0
