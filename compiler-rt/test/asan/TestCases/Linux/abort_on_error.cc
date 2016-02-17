// Check that with empty ASAN_OPTIONS, ASan reports on Linux don't crash
// the process (abort_on_error=0). See also Darwin/abort_on_error.cc.

// RUN: %clangxx_asan %s -o %t

// Intentionally don't inherit the default ASAN_OPTIONS.
// RUN: env ASAN_OPTIONS="" not %run %t 2>&1 | FileCheck %s
// When we use lit's default ASAN_OPTIONS, we shouldn't crash either. On Linux
// lit doesn't set ASAN_OPTIONS anyway.
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
}
