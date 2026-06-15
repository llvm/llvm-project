// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s

int main() {
  int count = 0;
  ++count; // !dex_label start
  ++count;
  ++count;
  ++count;
  return count;
} // !dex_label end

// CHECK: total_watched_steps: 5
// CHECK: correct_steps: 5
// CHECK: incorrect_steps: 0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: seen_values: 5
// CHECK: missing_values: 0

/*
---
!where {lines: !label start}:
    !value count: 0
!where {lines: !label start + 1}:
    !value count: 1
!where {lines: !label start+  2}:
    !value count: 2
!where {lines: !label start     +3}:
    !value count: 3
!where {lines: !label end-1}:
    !value count: 4
...
*/
