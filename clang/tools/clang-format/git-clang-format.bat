@echo off
where /q python
if not errorlevel 1 (
  python "%~dpn0" %*
) else (
  py -3 "%~dpn0" %*
)
