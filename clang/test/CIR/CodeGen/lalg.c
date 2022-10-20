// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o - | FileCheck %s

double dot() {
  double x = 0.0;
  double y = 0.0f;
  return x;
}

//      CHECK: %1 = cir.alloca f64, cir.ptr <f64>, ["x", init]
// CHECK-NEXT: %2 = cir.alloca f64, cir.ptr <f64>, ["y", init]
// CHECK-NEXT: %3 = cir.cst(0.000000e+00 : f64) : f64
// CHECK-NEXT: cir.store %3, %1 : f64, cir.ptr <f64>
// CHECK-NEXT: %4 = cir.cst(0.000000e+00 : f32) : f32
// CHECK-NEXT: %5 = cir.cast(floating, %4 : f32), f64
// CHECK-NEXT: cir.store %5, %2 : f64, cir.ptr <f64>
