// RUN: %clang_cc1 -std=c++23 -ast-print %s | FileCheck %s

// CHECK: void f(int x, int y) {
void f(int x, int y) {
  // CHECK-NEXT: {{\[}}[assume(true)]]
  [[assume(true)]];

  // CHECK-NEXT: {{\[}}[assume(2 + 4)]]
  [[assume(2 + 4)]];

  // CHECK-NEXT: {{\[}}[assume(x == y)]]
  [[assume(x == y)]];
}
