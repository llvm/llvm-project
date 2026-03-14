// PR24561
// RUN: %clangxx_msan -O2 -g %s -o %t && %run %t

#include <stdio.h>

struct A {
  int c1 : 7;
  int c8 : 1;
  int c9 : 1;
  A();
};

__attribute__((noinline)) A::A() : c8(1) {}

int main() {
  A* a = new A();
  if (a->c8 == 0)
    printf("zz\n");
  return 0;
}
