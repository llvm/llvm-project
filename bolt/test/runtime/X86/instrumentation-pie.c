/* Checks that BOLT correctly handles instrumentation of executables built
 * with PIE with further optimization.
 */
#include <stdio.h>

int foo(int x) { return x + 1; }

int fib(int x) {
  if (x < 2)
    return x;
  return fib(x - 1) + fib(x - 2);
}

int bar(int x) { return x - 1; }

int main(int argc, char **argv) {
  printf("fib(%d) = %d\n", argc, fib(argc));
  return 0;
}

/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -Wl,-q -pie -fpie

RUN: llvm-bolt %t.exe -instrument -instrumentation-file=%t.fdata \
RUN:   -o %t.instrumented

# Instrumented program needs to finish returning zero
RUN: %t.instrumented 1 2 3 | FileCheck %s -check-prefix=CHECK-OUTPUT

# Test that the instrumented data makes sense
RUN:  llvm-bolt %t.exe -o %t.bolted -data %t.fdata \
RUN:    -reorder-blocks=cache+ -reorder-functions=hfsort+

RUN: %t.bolted 1 2 3  | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: fib(4) = 3
*/
