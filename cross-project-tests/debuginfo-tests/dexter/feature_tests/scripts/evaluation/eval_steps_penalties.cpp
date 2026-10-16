// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

// Test evaluation of !step nodes in Dexter.

// CHECK: total_line_steps: 15
// CHECK: correct_line_steps: 11
// CHECK: correct_line_score: 73.3% (11/15)
// CHECK: misordered_line_steps: 2
// CHECK: missing_lines: 1
// CHECK: incorrect_line_steps: 2
// CHECK: unexpected_lines: 2

// We want some janky formatting for the sake of this test.
// clang-format off
int doublify(int N) { return N * 2; }

int stepBackwards(int N) {
  return doublify(            // !dex_label rbegin
      doublify(
        doublify(N))); // !dex_label rend
}

void reportError() {}

void pleasePassTrue(bool ShouldDefinitelyBeTrue = true) {
  if (!ShouldDefinitelyBeTrue) // !dex_label error_check
    reportError();
}

int factorial(int N) {
  int Result = 1; // !dex_label fac_start
  for (int I = N; I-- > 0;) {
    Result *= I;
  }
  if (Result > 0)
    return Result;
  return 0; // !dex_label fac_end
}

int main() {
  stepBackwards(10);
  pleasePassTrue(false);
  factorial(3);
  return 0;
}
// clang-format on

/*
---
!where {lines: !range [!label rbegin, !label rend]}:
  # Actual stepping order is reversed.
  # 3 steps, 1/3 correct, 2 misordered.
  !step at_least: [!label rbegin, !label rbegin + 1, !label rbegin + 2]
!where {lines: !range [!label error_check, !label error_check + 1]}:
  # "Never" line is stepped on.
  # 2 steps, 1/2 correct, 1 incorrect, 1 unexpected
  !step never: [!label error_check + 1]
!where {lines: !range [!label fac_start, !label fac_end]}:
  # Loop iterates 4 times instead of 3, and exits on line 39 instead of 38.
  # 10 steps, 9/10 correct, 1 incorrect, 1 unexpected, 1 missing
  !step exactly: [!label fac_start,
    !label fac_start + 1, !label fac_start + 2,
    !label fac_start + 1, !label fac_start + 2,
    !label fac_start + 1, !label fac_start + 2,
    !label fac_start + 1, !label fac_start + 4, !label fac_start + 5]
...
*/
