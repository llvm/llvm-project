// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %clang -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Ensure that an intercepted call in a [[clang::nonblocking]] function
//         is flagged as an error. Basic smoke test.

#include <stdio.h>
#include <stdlib.h>

void violation() [[clang::nonblocking]] {
  void *ptr = malloc(2);
  printf("ptr: %p\n", ptr); // ensure we don't optimize out the malloc
}

int main() {
  violation();
  return 0;
  // CHECK: Real-time violation: intercepted call to real-time unsafe function `malloc` in real-time context! Stack trace:
  // CHECK-NEXT: {{.*malloc*}}
}
