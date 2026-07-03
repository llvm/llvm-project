// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>
#include <stdlib.h>

// Violation reported in https://github.com/llvm/llvm-project/issues/86685.
void foo(int *s, float *f, long n) {
  for (long i = 0; i < n; ++i) {
    *f = 2;
    if (i == 1)
      break;

    // CHECK:      TypeSanitizer: type-aliasing-violation on address
    // CHECK-NEXT: WRITE of size 4 at {{.+}} with type int accesses an existing object of type float
    // CHECK-NEXT:   #0 {{.+}} in foo {{.*/?}}violation-pr86685.c:17
    *s = 4;
  }
}

int main(void) {
  union {
    int s;
    float f;
  } u = {0};
  foo(&u.s, &u.f, 2);
  printf("%.f\n", u.f);
  return 0;
}
