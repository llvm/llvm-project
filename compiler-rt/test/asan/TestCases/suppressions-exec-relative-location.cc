// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// If the executable is started from a different location, we should still
// find the suppression file located relative to the location of the executable.
// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: %clangxx_asan -O0 %s -o %t-dir/exec
// RUN: echo "interceptor_via_fun:crash_function" > \
// RUN:   %t-dir/supp.txt
// RUN: %env_asan_opts=suppressions='"supp.txt"' \
// RUN:   %run %t-dir/exec 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-IGNORE %s
// RUN: rm -rf %t-dir

// If the wrong absolute path is given, we don't try to construct
// a relative path with it.
// RUN: %env_asan_opts=suppressions='"/absolute/path"' not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-WRONG-FILE-NAME %s

// Test that we reject directory as filename.
// RUN: %env_asan_opts=suppressions='"folder/only/"' not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-WRONG-FILE-NAME %s

// XFAIL: android
// XFAIL: windows-msvc
// UNSUPPORTED: ios

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
// CHECK-IGNORE-NOT: AddressSanitizer: heap-buffer-overflow
// CHECK-IGNORE: ignored
// CHECK-WRONG-FILE-NAME: failed to read suppressions file
