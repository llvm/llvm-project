// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --binary %t -- %s | FileCheck %s

// Test evaluation of a simple Dexter test.

// CHECK: basic_evaluate.cpp:
// CHECK: total_watched_steps: 6
// CHECK: correct_steps: 4
// CHECK: incorrect_steps: 2
// CHECK: missing_var_steps: 1
// CHECK: unexpected_value_steps: 1
// CHECK: correct_step_coverage: 66.7% (4/6)
// CHECK: seen_values: 5
// CHECK: missing_values: 5

int multiply(int b, int a) {
    int result = a * b;
    return result;
}

int main() {
    int a = 6;
    int b = 7;
    int c = multiply(a, b);
    return c;
}

/*
---
!where {lines: 18}:
    !value a: 5 # 1 Incorrect, 1 Missing
    !value b: 6 # 1 Correct + Seen
    !value result: [40, 42] # 1 Correct + Seen, 1 Incorrect + Missing
!where {lines: 25}:
    !value a: [6, 6] # 1 Correct, 2 Seen
    !value b: 7 # 1 Correct + Seen
    !value not_real: 42 # 1 Incorrect + Missing
!where {lines: 100}: # Never entered
    !value irrelevant: 10 # 1 Missing
    !value unseen: 'abc' # 1 Missing
...
*/
