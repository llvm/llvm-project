// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --use-script --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} %S/Inputs/rewrite_expects_expected.cpp

/// Test that when we have a Dexter test with missing/unknown expected values,
/// Dexter produces a modified test file that is identical except for a modified
/// script section.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

// CHECK: Rewrote script to add 6 expected values.
// CHECK: Failed to rewrite 2 expected values.

// CHECK: total_watched_steps: 7
// CHECK: correct_steps: 6
// CHECK: incorrect_steps: 1
// CHECK: seen_values: 6
// CHECK: missing_values: 2

int multiply(int b, int a) {
  int result = a * b;
  return result; // !dex_label mul_ret
}

int main() {
  int a = 6;
  int b = 7;
  int c = multiply(a, b);
  return c; // !dex_label main_ret
}
// !dex_label never_reached
/*
---
? !where {lines: !label 'mul_ret'}
: !value 'a': '7'
  !value 'b': '6'
  !value 'result': '42'
? !where {lines: !label 'main_ret'}
: !value 'a': '6'
  !value 'b': '7'
  !value 'c': '42'
  !value 'not_real': null
? !where {lines: !label 'never_reached'}
: !value 'a': null
...
*/
