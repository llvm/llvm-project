// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Check that a function marked with [[clang::nonblocking]] cannot call a function that is blocking.

#include <stdio.h>
#include <stdlib.h>

// TODO: Remove when [[blocking]] is implemented.
extern "C" void __rtsan_notify_blocking_call(const char *function_name);

void custom_blocking_function() {
  // TODO: When [[blocking]] is implemented, don't call this directly.
  __rtsan_notify_blocking_call(__func__);
}

void safe_call() {
  // TODO: When [[blocking]] is implemented, don't call this directly.
  __rtsan_notify_blocking_call(__func__);
}

void process() [[clang::nonblocking]] { custom_blocking_function(); }

int main() {
  safe_call(); // This shouldn't die, because it isn't in nonblocking context.
  process();
  return 0;
  // CHECK-NOT: {{.*safe_call*}}
  // CHECK: ==ERROR: RealtimeSanitizer: blocking-call
  // CHECK-NEXT: Call to blocking function `custom_blocking_function` in real-time context!
  // CHECK-NEXT: {{.*custom_blocking_function*}}
  // CHECK-NEXT: {{.*process*}}
}
