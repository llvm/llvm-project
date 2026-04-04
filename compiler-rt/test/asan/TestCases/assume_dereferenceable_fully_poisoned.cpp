// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-assume-dereferenceable=1 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  char *p = (char *)malloc(10);
  free(p);
  // CHECK: AddressSanitizer: dereferencable-assumption-violation
  // CHECK: ASSUMPTION of size 10
  // CHECK-NEXT: range [0x{{.*}}, 0x{{.*}}) is NOT dereferenceable
  // CHECK-NOT: is dereferenceable
  __builtin_assume_dereferenceable(p, 10);
  return 0;
}
