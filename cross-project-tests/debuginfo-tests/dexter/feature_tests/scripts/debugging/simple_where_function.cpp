// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t -- %s | FileCheck %s

void assign(int *Target, int Value) {
    // A comment.
    *Target = Value;
}

void replace(int *Start, int Length, int Value, int NewValue) {
  int *Middle = Start + Length / 2;
  if (*Middle == Value) {
    assign(Middle, NewValue);
    return;
  }
  if (Length == 1)
    return;
  if (*Middle > Value)
    replace(Start, Length / 2, Value, NewValue);
  else
    replace(Middle + 1, (Length - 1) / 2, Value, NewValue);
}

int main() {
  int Array[] = {2, 4, 6, 8, 10};
  assign(Array + 1, 5);
  replace(Array, 5, 8, 9);
  return Array[1] + Array[3];
}

/// Test that we can use functions in !where nodes, and that Dexter steps
/// through the entirety of those functions. We expect both calls to `assign` to
/// be stepped through, but only the non-recursive call of `replace` should be
/// stepped through, as the !where matches to the rootmost applicable frame.

// CHECK:      assign
// CHECK-NEXT:   simple_where_function.cpp(6
// CHECK:        "Value": (int) 5

// CHECK:      replace
// CHECK-NEXT:   simple_where_function.cpp(10
// CHECK:        "Length": (int) 5
// CHECK:      replace
// CHECK-NEXT:   simple_where_function.cpp(11
// CHECK:        "Length": (int) 5
// CHECK:      replace
// CHECK-NEXT:   simple_where_function.cpp(15
// CHECK:        "Length": (int) 5
// CHECK:      replace
// CHECK-NEXT:   simple_where_function.cpp(17
// CHECK:        "Length": (int) 5
// CHECK:      replace
// CHECK-NEXT:   simple_where_function.cpp(20
// CHECK:        "Length": (int) 5

/// The recursive call should not be stepped through by Dexter.
// CHECK-NOT: "Length": (int) 2

/*
---
!where {function: assign}:
    !value Value: [5, 9]
!where {function: replace}:
    !value Length: 5
...
*/
