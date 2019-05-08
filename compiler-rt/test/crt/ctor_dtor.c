// RUN: %clang -fno-use-init-array -g -c %s -o %t.o
// RUN: %clang -fno-use-init-array -g -o %t -nostdlib %crt1 %crti %crtbegin %t.o -lc %libgcc %crtend %crtn
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>

// CHECK:      ctor()
// CHECK-NEXT: main()
// CHECK-NEXT: dtor()

void __attribute__((constructor)) ctor() {
  printf("ctor()\n");
}

void __attribute__((destructor)) dtor() {
  printf("dtor()\n");
}

int main() {
  printf("main()\n");
  return 0;
}
