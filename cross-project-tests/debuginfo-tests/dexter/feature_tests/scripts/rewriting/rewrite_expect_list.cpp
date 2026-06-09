// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --use-script --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} %S/Inputs/rewrite_expect_list_expected.cpp

/// Test that Dexter can write lists of expected values for simple scalar
/// variables.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

// CHECK: Rewrote script to add 3 expected values.

// CHECK: total_watched_steps: 90
// CHECK: correct_steps: 90
// CHECK: incorrect_steps: 0
// CHECK: seen_values: 86
// CHECK: missing_values: 0

int main() {
  int prev = 0;
  int current = 0;
  int next = 1;
  for (int i = 0; i < 30; ++i) {
    prev = current; // !dex_label loop
    current = next;
    next = prev + current;
  }
  return current;
}

/*
---
!where {lines: !label loop}:
    ? !value prev
    ? !value current
    ? !value next
...
*/
