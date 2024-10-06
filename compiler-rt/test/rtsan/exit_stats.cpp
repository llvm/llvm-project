// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: env RTSAN_OPTIONS="halt_on_error=false,print_stats_on_exit=true" %run %t 2>&1 | FileCheck %s

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
