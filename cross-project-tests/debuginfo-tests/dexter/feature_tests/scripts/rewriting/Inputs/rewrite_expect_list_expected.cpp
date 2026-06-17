// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} %S/Inputs/rewrite_expect_list_expected.cpp

/// Test that Dexter can write lists of expected values for simple scalar
/// variables.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.
///     Although we perform an exact file comparison, we use `diff` over `cmp`
///     for more legible lit output.

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
? !where {lines: !label 'loop'}
: !value 'prev':
  - '0'
  - '1'
  - '2'
  - '3'
  - '5'
  - '8'
  - '13'
  - '21'
  - '34'
  - '55'
  - '89'
  - '144'
  - '233'
  - '377'
  - '610'
  - '987'
  - '1597'
  - '2584'
  - '4181'
  - '6765'
  - '10946'
  - '17711'
  - '28657'
  - '46368'
  - '75025'
  - '121393'
  - '196418'
  - '317811'
  !value 'current':
  - '0'
  - '1'
  - '2'
  - '3'
  - '5'
  - '8'
  - '13'
  - '21'
  - '34'
  - '55'
  - '89'
  - '144'
  - '233'
  - '377'
  - '610'
  - '987'
  - '1597'
  - '2584'
  - '4181'
  - '6765'
  - '10946'
  - '17711'
  - '28657'
  - '46368'
  - '75025'
  - '121393'
  - '196418'
  - '317811'
  - '514229'
  !value 'next':
  - '1'
  - '2'
  - '3'
  - '5'
  - '8'
  - '13'
  - '21'
  - '34'
  - '55'
  - '89'
  - '144'
  - '233'
  - '377'
  - '610'
  - '987'
  - '1597'
  - '2584'
  - '4181'
  - '6765'
  - '10946'
  - '17711'
  - '28657'
  - '46368'
  - '75025'
  - '121393'
  - '196418'
  - '317811'
  - '514229'
  - '832040'
...
*/
