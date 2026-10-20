// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

/// Test that we correctly interpret nested !where+!and nodes during debugging
/// and evaluation.
/// With the conditional check, we should observe half the values of I ([0-3]
/// and [8-11]), and step into `walk` exactly 8 times.

void walk() {} // !dex_label walk

bool Red = false;
bool Green = true;

int main() {
  bool Light = Red;
  for (int I = 0; I < 16; ++I) {
    if (I % 8 == 0)
      Light = Green;
    else if (I % 4 == 0)
      Light = Red;
    // Light == Green from [0-3], [8-11]
    walk(); // !dex_label call
  }
  return 0;
}

// CHECK: total_watched_steps: 8
// CHECK: correct_steps: 8
// CHECK: incorrect_steps: 0
// CHECK: partial_step_correctness: 8.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: correct_step_coverage: 100.0% (8/8)
// CHECK: seen_values: 8
// CHECK: missing_values: 0
// CHECK: total_line_steps: 8
// CHECK: correct_line_steps: 8
// CHECK: correct_line_score: 100.0% (8/8)
// CHECK: misordered_line_steps: 0
// CHECK: missing_lines: 0
// CHECK: incorrect_line_steps: 0
// CHECK: unexpected_lines: 0

/*
---
!where {function: main}:
  !and {lines: !label call, conditions: 'Light == Green'}:
    !value I: [0, 1, 2, 3, 8, 9, 10, 11]
    !where {function: walk}:
      !step exactly: [!label walk, !label walk, !label walk, !label walk,
        !label walk, !label walk, !label walk, !label walk]
...
*/
