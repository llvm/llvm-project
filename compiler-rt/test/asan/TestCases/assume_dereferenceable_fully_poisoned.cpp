// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-assume-dereferenceable=1 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  char *p = (char *)malloc(10);
  free(p);
  // CHECK: AddressSanitizer: dereferenceable-assumption-violation
  // CHECK: ASSUME of size 10
  __builtin_assume_dereferenceable(p, 10);
  return 0;
}
