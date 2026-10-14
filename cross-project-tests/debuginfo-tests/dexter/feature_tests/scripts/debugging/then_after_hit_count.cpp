// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --skip-evaluate --binary %t \
// RUN:   -- %s | FileCheck %s

/// Test !then finish with !and{after_hit_count}.
/// The long loop will be exited when we hit the `!then finish` command,
/// which we will see after 101 hits of main: 50 hits on the loop line before
/// after_hit_count is reached, 50 hits from stepping off of the loop line, and
/// 1 hit from the step where we trigger the !then node.

// CHECK-LABEL:      Step 0
// CHECK-COUNT-101:   main
// CHECK-NOT: Step

int Until = 1000;
bool checkCows() { return --Until < 0; }

int main() {
  bool AreCowsHomeYet = false;
  while (!AreCowsHomeYet) {
    AreCowsHomeYet = checkCows(); // !dex_label loop
  }
  return 0;
}

/*
---
!where {lines: !label loop}:
    !value AreCowsHomeYet: false
    !and {after_hit_count: 50}: !then finish
...
*/
