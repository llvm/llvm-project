// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t \
// RUN:   -- %s | FileCheck %s

/// Test that we record hit counts for !where{function} nodes, even when there
/// are no other steps between each breakpoint hit.

/// We should only hit countdown 3 times, even though there is no gap between
/// the tail calls.
// CHECK-LABEL: Step 0
// CHECK-COUNT-3: countdown
// CHECK-NOT: countdown

/// All on one line for simplicity so that we only get one step per call.
// clang-format off
int countdown(int Num) { if (!Num) return 0; __attribute__((musttail)) return countdown(Num - 1); }
// clang-format on

int main() { return countdown(2) + countdown(2); }

/*
---
!where {function: countdown, for_hit_count: 3}:
  !value Num: [2, 1, 0]
...
*/
