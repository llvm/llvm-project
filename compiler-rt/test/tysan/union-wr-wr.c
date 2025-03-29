// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation

int main() {
  union {
    int i;
    short s;
  } u;

  u.i = 42;
  u.s = 1;

  printf("%d\n", u.i);
}
