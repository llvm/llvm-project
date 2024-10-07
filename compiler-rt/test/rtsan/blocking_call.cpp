// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Check that a function marked with [[clang::nonblocking]] cannot call a function that is blocking.

#include <stdio.h>
#include <stdlib.h>

void custom_blocking_function() [[clang::blocking]] {
  printf("In blocking function\n");
}

void safe_call() [[clang::blocking]] { printf("In safe call\n"); }

void process() [[clang::nonblocking]] { custom_blocking_function(); }

int main() {
  safe_call(); // This shouldn't die, because it isn't in nonblocking context.
  process();
  return 0;
  // CHECK: ==ERROR: RealtimeSanitizer: blocking-call
  // CHECK-NEXT: Call to blocking function `custom_blocking_function()` in real-time context!
  // CHECK-NEXT: {{.*custom_blocking_function*}}
  // CHECK-NEXT: {{.*process*}}

  // We should crash before this line is printed
  // CHECK-NOT: {{.*In blocking function.*}}

  // should only occur once
  // CHECK-NOT: ==ERROR: RealtimeSanitizer: blocking-call
}
