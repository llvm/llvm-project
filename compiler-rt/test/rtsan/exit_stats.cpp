// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %env_rtsan_opts="halt_on_error=false,print_stats_on_exit=true" %run %t 2>&1 | FileCheck %s
// RUN: %env_rtsan_opts="halt_on_error=true,print_stats_on_exit=true" not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-HALT

// UNSUPPORTED: ios

// Intent: Ensure exits stats are printed on exit.

#include <unistd.h>

void violation() [[clang::nonblocking]] {
  const int kNumViolations = 10;
  for (int i = 0; i < kNumViolations; i++) {
    usleep(1);
  }
}

int main() {
  violation();
  return 0;
}

// CHECK: RealtimeSanitizer exit stats:
// CHECK-NEXT: Total error count: 10
// CHECK-NEXT: Unique error count: 1

// CHECK-HALT: RealtimeSanitizer exit stats:
// CHECK-HALT-NEXT: Total error count: 1
// CHECK-HALT-NEXT: Unique error count: 1
