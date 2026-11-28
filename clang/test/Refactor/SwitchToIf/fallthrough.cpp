// RUN: clang-refactor -action switch-to-if %s -- 2>&1 | FileCheck %s

void g(int v) {
  switch (v) {
  case 3:
    alpha();
    [[fallthrough]];
  case 4:
    beta();
    break;
  default:
    gamma();
  }
}

// CHECK: if (v == 3) {
// CHECK-NEXT:     alpha();
// CHECK-NEXT:     beta();
// CHECK-NEXT: } else if (v == 4) {
// CHECK-NEXT:     beta();
// CHECK-NEXT: } else {
// CHECK-NEXT:     gamma();
// CHECK-NEXT: }
