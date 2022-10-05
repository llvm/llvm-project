// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o - | FileCheck %s

double dot() {
  double result = 0.0;
  return result;
}

//      CHECK: %1 = cir.alloca f64, cir.ptr <f64>, ["result", init]
// CHECK-NEXT: %2 = cir.cst(0.000000e+00 : f64) : f64
// CHECK-NEXT: cir.store %2, %1 : f64, cir.ptr <f64>
