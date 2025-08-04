// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

long foo(int *x, long *y) {
  *x = 0;
  *y = 1;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 8 at {{.*}} with type long accesses an existing object of type int
  // CHECK: {{#0 0x.* in foo .*int-long.c:}}[[@LINE-3]]

  return *x;
}

int main(void) {
  long l;
  printf("%ld\n", foo((int *)&l, &l));
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
