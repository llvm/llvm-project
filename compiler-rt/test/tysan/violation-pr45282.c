// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// https://github.com/llvm/llvm-project/issues/45282

#include <stdio.h>

int main(void) {

  double a[29], b[20];
  int i, j;

  for (i = 0; i < 20; ++i) {
    b[i] = 2.01f + 1.f;
    ((float *)a)[i] = 2.01f * 2.0145f;
    ((float *)a + 38)[i] = 2.01f * 1.0123f;
  }

  // CHECK:      TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: WRITE of size 8 at {{.+}} with type double accesses an existing object of type float
  // CHECK-NEXT:   in main {{.*/?}}violation-pr45282.c:25

  // loop of problems
  for (j = 2; j <= 4; ++j) {
    a[j - 1] = ((float *)a)[j] * ((float *)a + 38)[j - 1];
    ((float *)a + 38)[j - 1] = ((float *)a)[j - 1] + b[j - 1];
  }

  printf("((float *)a + 38)[2] = %f\n", ((float *)a + 38)[2]);

  return 0;
}
