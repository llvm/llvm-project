// RUN: %clangxx_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --implicit-check-not ERROR < %t.out

// Modified reproducer from https://github.com/llvm/llvm-project/issues/105960

#include <stdio.h>

struct inner1 {
  char buffer;
  int i;
};

struct inner2 {
  char buffer;
  int i;
  float endBuffer;
};

void init_inner1(inner1 *iPtr) { iPtr->i = 200; }
void init_inner2(inner2 *iPtr) {
  iPtr->i = 400;
  iPtr->endBuffer = 413.0f;
}

struct outer {
  inner1 foo;
  inner2 bar;
  char buffer;
};

int main(void) {
  outer *l = new outer();

  init_inner1(&l->foo);
  init_inner2(&l->bar);

  int access = l->foo.i;
  printf("Accessed value 1 is %d\n", access);
  access = l->bar.i;
  printf("Accessed value 2 is %d\n", access);
  float fAccess = l->bar.endBuffer;
  printf("Accessed value 3 is %f\n", fAccess);

  return 0;
}

// CHECK: Accessed value 1 is 200
// CHECK: Accessed value 2 is 400
// CHECK: Accessed value 3 is 413.0
