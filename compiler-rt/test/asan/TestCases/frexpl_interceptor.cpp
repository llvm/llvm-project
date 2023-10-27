// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Test the frexpl() interceptor.

// FIXME: MinGW-w64 implements `frexpl()` as a static import, so the dynamic
// interceptor seems to not work.
// XFAIL: target={{.*-windows-gnu}}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main() {
  long double x = 3.14;
  int *exp = (int *)malloc(sizeof(int));
  free(exp);
  double y = frexpl(x, exp);
  // CHECK: use-after-free
  // CHECK: SUMMARY
  return 0;
}
