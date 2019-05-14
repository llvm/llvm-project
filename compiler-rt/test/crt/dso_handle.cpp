// RUN: %clangxx -g -DCRT_SHARED -c %s -fPIC -o %tshared.o
// RUN: %clangxx -g -c %s -fPIC -o %t.o
// RUN: %clangxx -g -shared -o %t.so -nostdlib %crti %crtbegin %tshared.o %libstdcxx -lc -lm %libgcc %crtend %crtn
// RUN: %clangxx -g -o %t -nostdlib %crt1 %crti %crtbegin %t.o %libstdcxx -lc -lm %libgcc %t.so %crtend %crtn
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>

// CHECK: 1
// CHECK-NEXT: ~A()

#ifdef CRT_SHARED
bool G;
void C() {
  printf("%d\n", G);
}

struct A {
  A() { G = true; }
  ~A() {
    printf("~A()\n");
  }
};

A a;
#else
void C();

int main() {
  C();
  return 0;
}
#endif
