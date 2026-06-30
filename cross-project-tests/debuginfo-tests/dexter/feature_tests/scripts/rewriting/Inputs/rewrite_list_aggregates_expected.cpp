// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --use-script --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} \
// RUN:   %S/Inputs/rewrite_list_aggregates_expected.cpp

/// Test that Dexter can write expects for variables that are aggregates and
/// have more than one value, without writing any duplicate expected values.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

// CHECK: Rewrote script to add 1 expected values.

struct Point {
  int X;
  int Y;
};

int main() {
  Point P{1, 2};
  P.X = 3; // !dex_label start
  P.Y = 0;
  P.X = 1;
  P.Y = 2;
  P = {0, 0};
  return 0; // !dex_label end
}

// CHECK: total_watched_steps: 6
// CHECK: correct_steps: 6
// CHECK: incorrect_steps: 0
// CHECK: partial_step_correctness: 6.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: correct_step_coverage: 100.0% (6/6)
// CHECK: seen_values: 10
// CHECK: missing_values: 0

/*
---
? !where {lines: !range [!label 'start', !label 'end']}
: !value 'P':
  - X: '1'
    Y: '2'
  - X: '3'
    Y: '2'
  - X: '3'
    Y: '0'
  - X: '1'
    Y: '0'
  - X: '0'
    Y: '0'
...
*/
