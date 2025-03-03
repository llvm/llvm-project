// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// RUN: echo "interceptor_name:strlen" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// FIXME: Upload suppressions to device.
// XFAIL: android

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  char *a = (char *)malloc(6);
  free(a);
  size_t len = strlen(a); // BOOM
  fprintf(stderr, "strlen ignored, len = %zu\n", len);
}

// CHECK-CRASH: AddressSanitizer: heap-use-after-free
// CHECK-CRASH-NOT: strlen ignored
// CHECK-IGNORE-NOT: AddressSanitizer: heap-use-after-free
// CHECK-IGNORE: strlen ignored
