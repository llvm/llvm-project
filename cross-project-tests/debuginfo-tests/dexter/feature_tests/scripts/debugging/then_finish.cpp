// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t \
// RUN:   -- %s | FileCheck %s

/// Test that when we use !then finish, we finish the entire test immediately,
/// without observing any more steps afterwards.

void fizz() {}
void buzz() {}
void fizzbuzz() {}

void doFizzbuzz(int N) {
  // CHECK: then_finish.cpp([[# @LINE + 1 ]]:12)
  for (int I = 1; I < N; ++I) {
    // CHECK-COUNT-15: then_finish.cpp([[# @LINE + 1 ]]:9)
    if (I % 3 == 0) { // !dex_label loop_top
      if (I % 5 == 0)
        // CHECK: then_finish.cpp([[# @LINE + 1 ]]:9)
        fizzbuzz(); // !dex_label fizzbuzz
      else
        fizz();
    } else if (I % 5 == 0) {
      buzz();
    }
  }
}
/// We'll see main in "Frame 1" at the same step that we exit from; we should
/// not see it (or doFizzbuzz) afterwards.
// CHECK:      Frame 1:
// CHECK-NEXT:   main
// CHECK-NOT: main
// CHECK-NOT: doFizzbuzz

int main() {
  int V = 0;
  doFizzbuzz(30); // !dex_label main_start
  V = 1;
  return 0; // !dex_label main_end
}

/*
---
!where {function: main}:
    !where {function: doFizzbuzz}:
        !and {lines: !label loop_top}:
            !value I: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        !and {lines: !label fizzbuzz}: !then finish
    !and {lines: !range [!label main_start, !label main_end]}:
        !value V: [0, 1]
!where {lines: !label main_end}:
    !value V: 1
...
*/
