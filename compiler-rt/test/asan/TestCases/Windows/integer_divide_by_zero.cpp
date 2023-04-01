// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %env_asan_opts=handle_sigfpe=1 not %run %t 2>&1 | FileCheck %s

// Test the error output from dividing by zero.

#include <stdio.h>

int main() {
  puts("before ud2a");
  fflush(stdout);
  volatile int numerator = 1;
  volatile int divisor = 0;
  return numerator / divisor;
}
// CHECK: before ud2a
// CHECK: ERROR: AddressSanitizer: int-divide-by-zero on unknown address [[ADDR:0x[^ ]*]]
// CHECK-SAME: (pc [[ADDR]] {{.*}})
// CHECK-NEXT: #0 {{.*}} in main {{.*}}integer_divide_by_zero.cpp:{{.*}}
