// Check that the "unexpected format specifier" warning from the printf
// interceptor gives a backtrace

// RUN: %clang %s -Wno-format -Wno-format-invalid-specifier -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// The warning is only emitted by tools that use the common printf interceptor.
// UNSUPPORTED: ubsan, lsan, hwasan, msan, rtsan

#include <stdio.h>

__attribute__((noinline)) void log_it(const char *fmt, int v) {
  printf(fmt, v);
}

int main(void) {
  log_it("value=%y\n", 42);
  // CHECK: WARNING: unexpected format specifier in printf interceptor: %y
  // CHECK: {{ }}log_it
  // CHECK: {{ }}main
  printf("done\n");
  // CHECK: done
  return 0;
}
