// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %dexter_regression_test_cxx_build %s -o %t/test
// RUN: %dexter_regression_test_run --use-script --binary %t/test \
// RUN:   --results-directory %t/results -- %s 2>&1 | FileCheck %s
// RUN: diff %t/results/%{s:basename} \
// RUN:   %S/Inputs/rewrite_multiple_scripts_expected.cpp

/// Test that when a file contains more than one valid YAML script (but only one
/// Dexter script), the existing YAML is printed correctly.

/// NB: The exact contents of this file are compared against the expect file in
///     the Inputs/ directory; any changes to this file, including comments,
///     will require updating the corresponding expected file.

// CHECK: Rewrote script to add 1 expected values.

// CHECK: total_watched_steps: 1
// CHECK: correct_steps: 1
// CHECK: incorrect_steps: 0
// CHECK: seen_values: 1
// CHECK: missing_values: 0

/*
---
hr: # 1998 hr ranking
- Mark McGwire
- Sammy Sosa
# 1998 rbi ranking
rbi:
- Sammy Sosa
- Ken Griffey
...
*/

int main() {
  int ret = 0;
  return ret; // !dex_label ret
}

/*
---
? !where {lines: !label 'ret'}
: !value 'ret': '0'
...
*/
