// RUN: %clang_dsan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main(void) {
  char *p = malloc(16);
  free(p + 1);
  return 0;
}

// CHECK: ERROR: DoubleFreeSanitizer: invalid free on address
