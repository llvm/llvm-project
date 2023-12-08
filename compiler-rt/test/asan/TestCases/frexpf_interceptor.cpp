// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Test the frexpf() interceptor.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main() {
  float x = 3.14;
  int *exp = (int *)malloc(sizeof(int));
  free(exp);
  double y = frexpf(x, exp);
  // CHECK: use-after-free
  // CHECK: SUMMARY
  return 0;
}
