// RUN: clang-refactor -action switch-to-if %s -- 2>&1 | FileCheck %s

void test(int n) {
  switch (n) {
  case 1:
  case 2:
    handleSmall();
    break;
  case 10:
    handleLarge();
    break;
  default:
    handleOther();
  }
}

// CHECK: if (n == 1 || n == 2) {
// CHECK-NEXT:     handleSmall();
// CHECK-NEXT: } else if (n == 10) {
// CHECK-NEXT:     handleLarge();
// CHECK-NEXT: } else {
// CHECK-NEXT:     handleOther();
// CHECK-NEXT: }
