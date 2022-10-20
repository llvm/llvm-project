// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o - | FileCheck %s

double dot() {
  double x = 0.0;
  double y = 0.0f;
  double result = x * y;
  return result;
}

//      CHECK: %1 = cir.alloca f64, cir.ptr <f64>, ["x", init]
// CHECK-NEXT: %2 = cir.alloca f64, cir.ptr <f64>, ["y", init]
// CHECK-NEXT: %3 = cir.alloca f64, cir.ptr <f64>, ["result", init]
// CHECK-NEXT: %4 = cir.cst(0.000000e+00 : f64) : f64
// CHECK-NEXT: cir.store %4, %1 : f64, cir.ptr <f64>
// CHECK-NEXT: %5 = cir.cst(0.000000e+00 : f32) : f32
// CHECK-NEXT: %6 = cir.cast(floating, %5 : f32), f64
// CHECK-NEXT: cir.store %6, %2 : f64, cir.ptr <f64>
// CHECK-NEXT: %7 = cir.load %1 : cir.ptr <f64>, f64
// CHECK-NEXT: %8 = cir.load %2 : cir.ptr <f64>, f64
// CHECK-NEXT: %9 = cir.binop(mul, %7, %8) : f64
