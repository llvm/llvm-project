#include <stdio.h>

void foo(float a) {
  printf("a = %f\n", a);
}

typedef void (*fptr_t)(float);
fptr_t func = foo;

int main() {
  func(42);
  return 0;
}

/*
## Check that BOLT instrumentation runtime preserves xmm registers.

REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -fno-inline -Wl,-q
RUN: llvm-bolt %t.exe --instrument -o %t.instrumented
RUN: %t.instrumented | FileCheck %s

CHECK: a = 42.000000
*/
