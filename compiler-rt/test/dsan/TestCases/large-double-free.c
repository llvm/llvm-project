// RUN: %clang_dsan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main(void) {
  void *p = malloc(1 << 20);
  free(p);
  free(p);
  return 0;
}

// CHECK: ERROR: DoubleFreeSanitizer: double-free on address
// CHECK: First free of address
// CHECK: Original allocation of address
