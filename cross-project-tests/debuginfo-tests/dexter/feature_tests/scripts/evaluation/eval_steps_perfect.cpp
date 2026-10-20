// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s \
// RUN:   | FileCheck %s

// Test evaluation of !step nodes in Dexter.

// CHECK: total_line_steps: 150
// CHECK: correct_line_steps: 105
// CHECK: correct_line_score: 100.0% (105/105)
// CHECK: misordered_line_steps: 0
// CHECK: missing_lines: 0
// CHECK: incorrect_line_steps: 0
// CHECK: unexpected_lines: 0

void fizz() {}
void buzz() {}
void fizzbuzz() {}

void doFizzbuzz(int N) {
  for (int I = 1; I <= N; ++I) {
    if (I % 3 == 0) {
      if (I % 5 == 0)
        fizzbuzz();
      else
        fizz();
    } else if (I % 5 == 0) {
      buzz();
    }
  }
}

int main() {
  doFizzbuzz(10);
  return 0;
}

/*
---
!where {function: doFizzbuzz}:
  !step exactly: [20, 21, 26, 29, 20, 21, 26, 29, 20, 21, 22, 25, 26, 29, 20,
    21, 26, 29, 20, 21, 26, 27, 29, 20, 21, 22, 25, 26, 29, 20, 21, 26, 29, 20,
    21, 26, 29, 20, 21, 22, 25, 26, 29, 20, 21, 26, 27, 29, 20, 30]
  !step at_least: [25, 27, 25, 25, 27]
  !step never: [23]
...
*/
