@setlocal EnableDelayedExpansion

for %%I in (%SOURCE_INDEXES%) do (
  %PATHTOCLANGPP% -fuse-ld=lld -c !COMPILER_OPTIONS_%%I! !SOURCE_FILE_%%I! -o !OBJECT_FILE_%%I!
  if errorlevel 1 goto :FAIL
)

%PATHTOCLANGPP% -fuse-ld=lld %LINKER_OPTIONS% %OBJECT_FILES% -o %EXECUTABLE_FILE%
if errorlevel 1 goto :FAIL
goto :END

:FAIL
echo FAILED
exit /B 1

:END
exit /B 0
