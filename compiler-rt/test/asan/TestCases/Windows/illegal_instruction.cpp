// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %env_asan_opts=handle_sigill=1 not %run %t 2>&1 | FileCheck %s
// msvc doesn't have a __builtin_trap equivalent
// XFAIL: msvc

// Test the error output from an illegal instruction.

#include <stdio.h>

int main() {
  puts("before ud2a");
  fflush(stdout);
  __builtin_trap();
  return 0;
}
// CHECK: before ud2a
// CHECK: ERROR: AddressSanitizer: illegal-instruction on unknown address [[ADDR:0x[^ ]*]]
// CHECK-SAME: (pc [[ADDR]] {{.*}})
// CHECK-NEXT: #0 {{.*}} in main {{.*}}illegal_instruction.cpp:{{.*}}
