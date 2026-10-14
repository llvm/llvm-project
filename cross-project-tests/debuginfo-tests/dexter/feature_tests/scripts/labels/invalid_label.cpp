// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --source-root-dir %S/Inputs \
// RUN:   --binary %t -- %s 2>&1 | FileCheck %s

int main() {
  int a = 4;
  int b = 4;
  return b - a; // !dex_label unused
}

// CHECK:      error: Error with node: Label(used): Label "used" not found
// CHECK-SAME: in file "{{.*}}/invalid_label.cpp"

/*
---
!where {lines: !label used}:
    !value a: 4
    !value b: 4
...
*/
