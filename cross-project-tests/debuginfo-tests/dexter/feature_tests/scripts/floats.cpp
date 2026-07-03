// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s \
// RUN:   | FileCheck %s

/// Test that we correctly match float values against Float nodes, and also
/// correctly tally up the number of seen values for Float nodes that contain
/// expected value lists.

float slowNonInverseSquareRoot(float N) {
  float CurrentGuess = N * 0.5f;
  for (int I = 0; I < 8; ++I) { // !dex_label start
    float Complement = N / CurrentGuess;
    CurrentGuess = (CurrentGuess + Complement) * 0.5f; // !dex_label mid_loop
  }
  return CurrentGuess; // !dex_label end
}

int main() {
  slowNonInverseSquareRoot(50.0);
  slowNonInverseSquareRoot(100.0);
  return 0;
}

// CHECK: total_watched_steps: 94
// CHECK: correct_steps: 78
// CHECK: incorrect_steps: 16
// CHECK: partial_step_correctness: 78.0
// CHECK: missing_var_steps: 0
// CHECK: unexpected_value_steps: 16
// CHECK: correct_step_coverage: 83.0% (78/94)
// CHECK: seen_values: 17
// CHECK: missing_values: 7

/*
---
!where {function: slowNonInverseSquareRoot}:
  # All checks pass: 60 watched steps, 60 correct values, 12 seen values.
  !and {conditions: "N == 50.0"}:
    !and {lines: !range [!label start, !label end]}:
      !value CurrentGuess: !float
        values: [25, 13.5, 8.6, 7.2, 7.07]
        range: 0.01
      !value N: !float 50
    !and {lines: !label mid_loop}:
      !value Complement:
        - !float 2
        - !float 3.70 +- 0.01
        - !float 5.81 +- 0.01
        - !float 6.93 +- 0.01
        - !float 7.070 +- 0.001
        - !float 7.0710 +- 0.0001
  # Some checks fail: 34 watched steps, 18 correct values, 12 seen values.
  !and {conditions: "N == 100.0"}:
    !and {lines: !range [!label start, !label end]}:
      # 3 correct, 2 unseen, 4 unexpected
      !value CurrentGuess: !float [50, 26, 15, 11, 10]
    !and {lines: !label mid_loop}:
      # 3 correct, 4 unseen, 4 unexpected
      !value Complement:
        - !float 2 +- 0.0001      # Correct
        - !float 3.84 +- 0.0001   # Incorrect, actual=3.84615374
        - !float 6.70 +- 0.0001   # Incorrect, actual=6.70103121
        - !float 9.24 +- 0.0001   # Incorrect, actual=9.24893665
        - !float 9.96 +- 0.0001   # Incorrect, actual=9.96959781
        - !float 9.99 +- 0.0001   # Correct
        - !float 10.0 +- 0.0001   # Correct
...
*/
