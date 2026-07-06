// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} \
// RUN:   %S/Inputs/rewrite_scopes_expected.cpp

/// Tests that we can collect the values of all available variables with
/// !value/all and produce corresponding variable expects.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

char There[] = "Here";

int main() {
  int One = 2;
  char Red[] = "Blue";
  return 0; // !dex_label ret
}

// CHECK: total_watched_steps: 3
// CHECK: correct_steps: 3
// CHECK: incorrect_steps: 0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 0
// CHECK: seen_values: 11
// CHECK: missing_values: 0

/*
---
!where {lines: !label ret}:
    ? !value/all Locals
    ? !value/all Globals
    # Invalid scopes won't appear in the output.
    ? !value/all NotARealScope
...
*/
