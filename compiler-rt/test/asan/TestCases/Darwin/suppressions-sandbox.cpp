// Compile the intermediate function to a dylib without -fsanitize to avoid
// suppressing symbols in sanitized code.
// RUN: %clangxx -O0 -DSHARED_LIB %s -dynamiclib -o %t.dylib -framework Foundation

// Check that without suppressions, we catch the issue.
// RUN: %clangxx_asan -O0 %s -o %t -framework Foundation %t.dylib
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s

// Check that suppressing a function name works within a no-fork sandbox
// RUN: echo "interceptor_via_fun:createCFString" > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' \
// RUN:   sandbox-exec -p '(version 1)(allow default)(deny process-fork)' \
// RUN:   %run %t 2>&1 | FileCheck --check-prefix=CHECK-IGNORE %s

// sandbox-exec isn't available on iOS
// UNSUPPORTED: ios

// Symbolizer fails to find test functions on current macOS bot version
// XFAIL: target=arm{{.*}}

#include <CoreFoundation/CoreFoundation.h>

#if defined(SHARED_LIB)

extern "C" {
// Disable optimizations to ensure that this function appears on the stack trace so our
// configured suppressions `interceptor_via_fun:createCFString` can take effect.
__attribute__((disable_tail_calls)) CFStringRef
createCFString(const unsigned char *bytes, CFIndex length) {
  return CFStringCreateWithBytes(kCFAllocatorDefault, bytes, length,
                                 kCFStringEncodingUTF8, FALSE);
}
}

#else

extern "C" {
CFStringRef createCFString(const unsigned char *bytes, CFIndex length);
}

int main() {
  char *a = (char *)malloc(6);
  strcpy(a, "hello");
  // Intentional out-of-bounds access that will be caught unless an ASan suppression is provided.
  CFStringRef str = createCFString((unsigned char *)a, 10); // BOOM
  // If this is printed to stderr then the ASan suppression has worked.
  fprintf(stderr, "Ignored.\n");
  free(a);
  CFRelease(str);
}

#endif

// CHECK-CRASH: AddressSanitizer: heap-buffer-overflow
// CHECK-CRASH-NOT: Ignored.
// CHECK-IGNORE-NOT: AddressSanitizer: heap-buffer-overflow
// CHECK-IGNORE: Ignored.
