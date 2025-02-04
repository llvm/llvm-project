// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// RUN: echo "alloc_dealloc_mismatch:function" > %t.supp
// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=alloc_dealloc_mismatch=1:suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s
// RUN: %clangxx_asan -O3 %s -o %t && %env_asan_opts=alloc_dealloc_mismatch=1:suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// FIXME: Upload suppressions to device.
// XFAIL: android

// FIXME: atos does not work for inlined functions, yet llvm-symbolizer
// does not always work with debug info on Darwin.
// UNSUPPORTED: darwin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void function() {
  char *a = (char *)malloc(6);
  a[0] = '\0';
  size_t len = strlen(a);
  delete a; // BOOM
  fprintf(stderr, "strlen ignored, len = %zu\n", len);
}

int main() { function(); }

// CHECK-CRASH: AddressSanitizer: alloc-dealloc-mismatch
// CHECK-CRASH-NOT: strlen ignored
// CHECK-IGNORE-NOT: AddressSanitizer: alloc-dealloc-mismatch
// CHECK-IGNORE: strlen ignored
