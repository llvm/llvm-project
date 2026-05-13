// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Test the frexpl() interceptor.

// FIXME: MinGW-w64 implements `frexpl()` as a static import, so the dynamic
// interceptor seems to not work.
// XFAIL: target={{.*-windows-gnu}}

// clang will expand frexpl to a function(with mangle name) that calls frexp.
// On AIX, frexp can not be intercepted.
// UNSUPPORTED: target={{.*aix.*}}

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
