// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t -- %s | FileCheck %s

/// Test that when we use !then step_out, we jump out of the current frame, but
/// continue stepping through the frame above.

void fizz() {}
void buzz() {}
void fizzbuzz() {}

void doFizzbuzz(int N) {
// CHECK: then_step_out.cpp([[# @LINE + 1 ]]:14)
    for (int I = 1; I < N; ++I) {
// CHECK-COUNT-15: then_step_out.cpp([[# @LINE + 1 ]]:13)
        if (I % 3 == 0) {  // !dex_label loop_top
            if (I % 5 == 0)
// CHECK: then_step_out.cpp([[# @LINE + 1 ]]:17)
                fizzbuzz(); // !dex_label fizzbuzz
            else
                fizz();
        } else if (I % 5 == 0) {
            buzz();
        }
    }
}
// CHECK-NOT: doFizzbuzz

int main() {
  int V = 0;
  doFizzbuzz(30); // !dex_label main_start
  V = 1;
// CHECK: then_step_out.cpp([[# @LINE + 1 ]]:3)
  return 0; // !dex_label main_end
}

/*
---
!where {function: main}:
    !where {function: doFizzbuzz}:
        !and {lines: !label loop_top}:
            !value I: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        !and {lines: !label fizzbuzz}: !then step_out
    !and {lines: !range [!label main_start, !label main_end]}:
        !value V: [0, 1]
...
*/
