// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-assume-dereferenceable=1 -fsanitize-recover=address %s -o %t
// RUN: %env_asan_opts=halt_on_error=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER

#include <stdio.h>
#include <stdlib.h>

int main() {
  char *p = (char *)malloc(10);

  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR:0x[0-9a-fA-F]+]]

  // CHECK-RECOVER: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR:0x[0-9a-fA-F]+]]
  __builtin_assume_dereferenceable(p, 20);
  free(p);

  fprintf(stderr, "EXECUTED AFTER ERROR\n");
  // CHECK-NOT: EXECUTED AFTER ERROR
  // CHECK-RECOVER: EXECUTED AFTER ERROR

  return 0;
}
