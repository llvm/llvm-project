// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t -framework Foundation
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// Check that suppressing a function name works within a no-fork sandbox
// RUN: echo "interceptor_via_fun:createCFString" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"':verbostiy=1:detect_leaks=0 \
// RUN:   sandbox-exec -p '(version 1)(allow default)(deny process-fork)' \
// RUN:   %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// sandbox-exec isn't available on iOS
// UNSUPPORTED: ios

#include <CoreFoundation/CoreFoundation.h>

// Use a noinline wrapper function to suppress to stabilize test.
__attribute__((noinline)) CFStringRef createCFString(const unsigned char *bytes,
                                                     CFIndex length) {
  return CFStringCreateWithBytes(kCFAllocatorDefault, bytes, length,
                                 kCFStringEncodingUTF8, FALSE);
}

int main() {
  char *a = (char *)malloc(6);
  strcpy(a, "hello");
  CFStringRef str = createCFString((unsigned char *)a, 10); // BOOM
  fprintf(stderr, "Ignored.\n");
  free(a);
  CFRelease(str);
}

// CHECK-CRASH: AddressSanitizer: heap-buffer-overflow
// CHECK-CRASH-NOT: Ignored.
// CHECK-IGNORE-NOT: AddressSanitizer: heap-buffer-overflow
// CHECK-IGNORE: Ignored.
