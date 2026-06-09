// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --use-script --skip-evaluate --binary %t -- %s | FileCheck %s

char There[] = "Here";

int main() {
  int One = 2;
  char Red[] = "Blue";
  return 0; // !dex_label ret
}

/// Test that we can use functions in !where nodes, and that Dexter steps
/// through the entirety of those functions. We expect both calls to `assign` to
/// be stepped through, but only the non-recursive call of `replace` should be
/// stepped through, as the !where matches to the rootmost applicable frame.

// CHECK:      Step 0
// CHECK:          main
// CHECK:      Variable Scopes:
// CHECK-NEXT:   Globals: [::There]
// CHECK-NEXT:   Locals: [One, Red]
// CHECK-NEXT: Variables:
// CHECK-NEXT:   "::There": (char[5]) "Here"
// CHECK-NEXT:     "[0]": (char) 'H'
// CHECK-NEXT:     "[1]": (char) 'e'
// CHECK-NEXT:     "[2]": (char) 'r'
// CHECK-NEXT:     "[3]": (char) 'e'
// CHECK-NEXT:     "[4]": (char) '\0'
// CHECK-NEXT:   "One": (int) 2
// CHECK-NEXT:   "Red": (char[5]) "Blue"
// CHECK-NEXT:     "[0]": (char) 'B'
// CHECK-NEXT:     "[1]": (char) 'l'
// CHECK-NEXT:     "[2]": (char) 'u'
// CHECK-NEXT:     "[3]": (char) 'e'
// CHECK-NEXT:     "[4]": (char) '\0'

// CHECK-NOT: Step 1

/*
---
!where {lines: !label ret}:
    ? !value/all Locals
    ? !value/all Globals
    # Invalid scopes won't appear in the output.
    ? !value/all NotARealScope
...
*/
