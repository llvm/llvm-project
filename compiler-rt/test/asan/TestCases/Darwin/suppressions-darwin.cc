// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t -framework Foundation
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// Check that suppressing the interceptor by name works.
// RUN: echo "interceptor_name:memmove" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// Check that suppressing by interceptor name works even without the symbolizer
// RUN: %env_asan_opts=suppressions='"%t.supp"':symbolize=false %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// Check that suppressing all reports from a library works.
// RUN: echo "interceptor_via_lib:CoreFoundation" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// Check that suppressing library works even without the symbolizer.
// RUN: %env_asan_opts=suppressions='"%t.supp"':symbolize=false %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

#include <CoreFoundation/CoreFoundation.h>

int main() {
  char *a = (char *)malloc(6);
  strcpy(a, "hello");
  CFStringRef str =
      CFStringCreateWithBytes(kCFAllocatorDefault, (unsigned char *)a, 10,
                              kCFStringEncodingUTF8, FALSE); // BOOM
  fprintf(stderr, "Ignored.\n");
  free(a);
}

// CHECK-CRASH: AddressSanitizer: heap-buffer-overflow
// CHECK-CRASH-NOT: Ignored.
// CHECK-IGNORE-NOT: AddressSanitizer: heap-buffer-overflow
// CHECK-IGNORE: Ignored.
