:: REQUIRES: x86, system-windows
:: RUN: echo | llvm-mc -filetype=obj -triple=x86_64-unknown-linux - -o %t0.o
:: RUN: not ld.lld -o /dev/null -lidontexist --error-handling-script=%s %t0.o 2>&1 |\
:: RUN:   FileCheck --check-prefix=CHECK-LIB %s
:: RUN: not ld.lld -o /dev/null -lidontexist --error-handling-script=%s.nope %t0.o 2>&1 |\
:: RUN:   FileCheck --check-prefix=CHECK-SCRIPT-DOES-NOT-EXIST -DFILE=%s.nope %s
::
:: CHECK-LIB:      script: info: called with missing-lib idontexist
:: CHECK-LIB-NEXT: error: unable to find library -lidontexist

:: CHECK-SCRIPT-DOES-NOT-EXIST:      error: unable to find library -lidontexist
:: CHECK-SCRIPT-DOES-NOT-EXIST-NEXT: error: error handling script '[[FILE]]' failed to execute

@echo off
echo "script: info: called with %*"
