// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t \
// RUN:   -- %s | FileCheck %s

/// Check that the debugger is able to fetch the components of aggregate values.

struct Point {
  int X;
  int Y;
  int Z;
};

struct Rect {
  Point TopLeft;
  Point BottomRight;
};

int main() {
  Point P{1, 2, 3};
  int *I = &P.X;
  Rect R{{1, 1, 1}, {2, 2, 2}};
  int L[] = {0, 1, 2, 3, 4};
  return 0; // !dex_label ret
}

// CHECK:       Frame 0:
// CHECK-NEXT:    main
// CHECK:       "I": (int *)
// CHECK-NEXT:    "*I": (int) 1
// CHECK-NEXT:  "L": (int[5])
// CHECK-NEXT:    "[0]": (int) 0
// CHECK-NEXT:    "[1]": (int) 1
// CHECK-NEXT:    "[2]": (int) 2
// CHECK-NEXT:    "[3]": (int) 3
// CHECK-NEXT:    "[4]": (int) 4
// CHECK-NEXT:  "P": (Point)
// CHECK-NEXT:    "X": (int) 1
// CHECK-NEXT:    "Y": (int) 2
// CHECK-NEXT:    "Z": (int) 3
// CHECK-NEXT:  "R": (Rect)
// CHECK-NEXT:    "TopLeft": (Point)
// CHECK-NEXT:      "X": (int) 1
// CHECK-NEXT:      "Y": (int) 1
// CHECK-NEXT:      "Z": (int) 1
// CHECK-NEXT:    "BottomRight": (Point)
// CHECK-NEXT:      "X": (int) 2
// CHECK-NEXT:      "Y": (int) 2
// CHECK-NEXT:      "Z": (int) 2

/*
---
!where {lines: !label ret}:
    !value P:
        X: 1
        Y: 2
        Z: 3
    !value I:
        "*I": 1
    !value R:
        TopLeft:
            X: 1
            Y: 1
            Z: 1
        BottomRight:
            X: 2
            Y: 2
            Z: 2
    !value L:
        "[0]": 0
        "[1]": 1
        "[2]": 2
        "[3]": 3
        "[4]": 4
...
*/
