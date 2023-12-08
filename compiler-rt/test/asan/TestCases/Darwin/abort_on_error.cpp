// Check that with empty ASAN_OPTIONS, ASan reports on OS X actually crash
// the process (abort_on_error=1). See also Linux/abort_on_error.cpp.

// RUN: %clangxx_asan %s -o %t

// Intentionally don't inherit the default ASAN_OPTIONS.
// RUN: env ASAN_OPTIONS="" not --crash %run %t 2>&1 | FileCheck %s
// When we use lit's default ASAN_OPTIONS, we shouldn't crash.
// RUN: not %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
}
