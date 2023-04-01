// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %env_asan_opts=handle_sigill=1 not %run %t 2>&1 | FileCheck %s

// Test the error output from a breakpoint. Assertion-like macros often end in
// int3 on Windows.

#include <stdio.h>

int main() {
  puts("before breakpoint");
  fflush(stdout);
  __debugbreak();
  return 0;
}
// CHECK: before breakpoint
// CHECK: ERROR: AddressSanitizer: breakpoint on unknown address [[ADDR:0x[^ ]*]]
// CHECK-SAME: (pc [[ADDR]] {{.*}})
// CHECK-NEXT: #0 {{.*}} in main {{.*}}breakpoint.cpp:{{.*}}
