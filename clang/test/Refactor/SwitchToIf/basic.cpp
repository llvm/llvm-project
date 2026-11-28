// RUN: clang-refactor -action switch-to-if %s -- 2>&1 | FileCheck %s

void foo(int x) {
  switch (x) { // CHECK: Start refactoring here
  case 1:
    bar();
    break;
  case 2:
    baz();
    break;
  default:
    qux();
  }
}

// CHECK: if (x == 1) {
// CHECK-NEXT:     bar();
// CHECK-NEXT: } else if (x == 2) {
// CHECK-NEXT:     baz();
// CHECK-NEXT: } else {
// CHECK-NEXT:     qux();
// CHECK-NEXT: }
