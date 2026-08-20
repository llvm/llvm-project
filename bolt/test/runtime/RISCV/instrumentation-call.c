/* Checks that BOLT correctly handles instrumentation of executables.

REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -Wl,-q

RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
RUN:   -o %t.instrumented

# Instrumented program needs to finish returning zero
RUN: %t.instrumented 1 2 3 | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: fib(4) = 3
*/

#include <stdio.h>

int fib(int x) {
  if (x < 2)
    return x;
  return fib(x - 1) + fib(x - 2);
}

int main(int argc, char **argv) {
  printf("fib(%d) = %d\n", argc, fib(argc));
  return 0;
}
