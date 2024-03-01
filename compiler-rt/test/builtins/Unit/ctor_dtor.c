// REQUIRES: crt

// RUN: %clang -fno-use-init-array -g -c %s -o %t.o
// RUN: %clang -o %t -no-pie -nostdlib %crt1 %crti %crtbegin %t.o -lc %libgcc %crtend %crtn
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

// Ensure the various startup functions are called in the proper order.

/// ctor() is here if ld.so/libc supports DT_INIT/DT_FINI
// CHECK:      main()
/// dtor() is here if ld.so/libc supports DT_INIT/DT_FINI

struct object;
static int counter;

void __attribute__((constructor)) ctor() {
  printf("ctor()\n");
  ++counter;
}

void __attribute__((destructor)) dtor() {
  printf("dtor()\n");
  if (--counter != 0)
    abort();
}

int main() {
  printf("main()\n");
  return 0;
}
