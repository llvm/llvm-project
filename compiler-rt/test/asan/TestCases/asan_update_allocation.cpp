// RUN: %clangxx_asan -O0 -DSIZE=10 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O0 -DSIZE=10000000 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/asan_interface.h>

void UPDATE(void *p) {
  __asan_update_allocation_context(p);
}

int main() {
  char *x = (char*)malloc(SIZE * sizeof(char));
  UPDATE(x);
  free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK: UPDATE
}
