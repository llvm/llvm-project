// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: env RTSAN_OPTIONS="halt_on_error=false,print_stats_on_exit=true" %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

// Intent: Ensure all errors are deduplicated.

#include <unistd.h>

const int kNumViolations = 10;

void violation() [[clang::nonblocking]] {
  for (int i = 0; i < kNumViolations; i++)
    usleep(1);
}

void violation2() [[clang::nonblocking]] {
  for (int i = 0; i < kNumViolations; i++)
    violation();
}

void double_violation() [[clang::nonblocking]] {
  violation();
  violation2();
}

int main() {
  violation();        // 1 unique errors here, but 10 total
  violation2();       // 1 unique errors here, but 100 total
  double_violation(); // 2 unique errors here, but 110 total
  return 0;
}

// CHECK-COUNT-4: ==ERROR:
// CHECK-NOT: ==ERROR:

// CHECK: RealtimeSanitizer exit stats:
// CHECK-NEXT: Total error count: 220
// CHECK-NEXT: Unique error count: 4
