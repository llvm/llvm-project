// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

/// Test that Dexter respects for and after hit_counts.

void receive(int N) {}

int main() {
  int Current = 0;
  int Increment = 1;
  for (int I = 0; I < 10; ++I) {
    receive(Current); // !dex_label loop
    Current += Increment++;
  }
  return 0;
}

// CHECK: total_watched_steps: 10
// CHECK: correct_steps: 10
// CHECK: incorrect_steps: 0
// CHECK: partial_step_correctness: 10.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: correct_step_coverage: 100.0% (10/10)
// CHECK: seen_values: 10
// CHECK: missing_values: 0

/*
---
!where {function: receive, after_hit_count: 3, for_hit_count: 4}:
  !value N: [6, 10, 15, 21]
!where {lines: !label loop, for_hit_count: 3}:
  !value Current: [0, 1, 3]
!where {lines: !label loop, after_hit_count: 7}:
  !value Current: [28, 36, 45]
...
*/
