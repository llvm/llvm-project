// Copy of instrumentation-ind-call.c, but checking that BOLT refuses the
// instrumentation because of BTI.
// TODO: once instrumentation support for BTI is added, update this to check the
// same as instrumentation-ind-call.c

#include <stdio.h>

typedef int (*func_ptr)(int, int);

int add(int a, int b) { return a + b; }

int main() {
  func_ptr fun;
  fun = add;
  int sum = fun(10, 20); // indirect call to 'add'
  printf("The sum is: %d\n", sum);
  return 0;
}
/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -Wl,-q -no-pie -fpie \
RUN: -mbranch-protection=standard -Wl,-z,force-bti

RUN: not llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
RUN:   -o %t.instrumented 2>&1 | FileCheck %s

CHECK: binary is using BTI
CHECK: FATAL BOLT-ERROR: instrumenting binaries using BTI is not supported
*/
