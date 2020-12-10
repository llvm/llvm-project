// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// RUN: echo "interceptor_via_fun:crash_function" > %t.supp
// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s
// RUN: %clangxx_asan -O3 %s -o %t && %env_asan_opts=suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// XFAIL: android
// UNSUPPORTED: ios

// FIXME: atos does not work for inlined functions, yet llvm-symbolizer
// does not always work with debug info on Darwin.
// UNSUPPORTED: darwin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void crash_function() {
  char *a = (char *)malloc(6);
  free(a);
  size_t len = strlen(a); // BOOM
  fprintf(stderr, "strlen ignored, len = %zu\n", len);
}

int main() {
  crash_function();
}

// CHECK-CRASH: AddressSanitizer: heap-use-after-free
// CHECK-CRASH-NOT: strlen ignored
// CHECK-IGNORE-NOT: AddressSanitizer: heap-use-after-free
// CHECK-IGNORE: strlen ignored
