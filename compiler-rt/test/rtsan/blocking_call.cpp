// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Check that a function marked with [[clang::nonblocking]] cannot call a function that is blocking.

#include <stdio.h>
#include <stdlib.h>

void custom_blocking_function() [[clang::blocking]] {
  printf("In blocking function\n");
}

void realtime_function() [[clang::nonblocking]] { custom_blocking_function(); }
void nonrealtime_function() { custom_blocking_function(); }

int main() {
  nonrealtime_function();
  realtime_function();
  return 0;
}

// CHECK: ==ERROR: RealtimeSanitizer: blocking-call
// CHECK-NEXT: Call to blocking function `custom_blocking_function()` in real-time context!
// CHECK-NEXT: {{.*custom_blocking_function*}}
// CHECK-NEXT: {{.*realtime_function*}}

// should only occur once
// CHECK-NOT: ==ERROR: RealtimeSanitizer: blocking-call
