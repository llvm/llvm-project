// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --skip-evaluate --binary %t \
// RUN:   -- %s | FileCheck %s

/// Test !where nodes work with for_hit_count.

/// All on one line for simplicity so that we only get one step per call.
int collatz(int N) { return (N % 2) ? N * 3 + 1 : N / 2; }

int main() {
  int MaxAttempts = 50;
  int Value = 472959593;
  for (int I = 0; I < MaxAttempts; ++I) {
    Value = collatz(Value); // !dex_label loop_start
    if (Value == 1)
      break; // !dex_label loop_end
  }
  return Value == 1 ? 0 : 1;
}

// CHECK:      Step 0
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(14:21)
// CHECK:      Step 1
// CHECK:        Frame 0:
// CHECK-NEXT:     collatz(int)
// CHECK-NEXT:     where_for_hit_count.cpp(8:30)
// CHECK:      Step 2
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(14:11)
// CHECK:      Step 3
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(15:15)
// CHECK:      Step 4
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(17:3)
// CHECK:      Step 5
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(14:21)
// CHECK:      Step 6
// CHECK:        Frame 0:
// CHECK-NEXT:     collatz(int)
// CHECK-NEXT:     where_for_hit_count.cpp(8:30)
// CHECK:      Step 7
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(14:11)
// CHECK:      Step 8
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(15:15)
// CHECK:      Step 9
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(17:3)
// CHECK:      Step 10
// CHECK:        Frame 0:
// CHECK-NEXT:     collatz(int)
// CHECK-NEXT:     where_for_hit_count.cpp(8:30)
// CHECK:      Step 11
// CHECK:        Frame 0:
// CHECK-NEXT:     main
// CHECK-NEXT:     where_for_hit_count.cpp(14:11)

/*
---
!where {function: collatz, for_hit_count: 3}:
    !value N: 0
!where {lines: !range [!label loop_start, !label loop_end], for_hit_count: 2}:
    !value Value: 0
    !value I: 0
...
*/
