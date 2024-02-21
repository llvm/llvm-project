// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o - | FileCheck %s

double dot() {
  double x = 0.0;
  double y = 0.0f;
  double result = x * y;
  return result;
}

//      CHECK: %1 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["x", init]
// CHECK-NEXT: %2 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["y", init]
// CHECK-NEXT: %3 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["result", init]
// CHECK-NEXT: %4 = cir.const(#cir.fp<0.000000e+00> : !cir.double) : !cir.double
// CHECK-NEXT: cir.store %4, %1 : !cir.double, cir.ptr <!cir.double>
// CHECK-NEXT: %5 = cir.const(#cir.fp<0.000000e+00> : !cir.float) : !cir.float
// CHECK-NEXT: %6 = cir.cast(floating, %5 : !cir.float), !cir.double
// CHECK-NEXT: cir.store %6, %2 : !cir.double, cir.ptr <!cir.double>
// CHECK-NEXT: %7 = cir.load %1 : cir.ptr <!cir.double>, !cir.double
// CHECK-NEXT: %8 = cir.load %2 : cir.ptr <!cir.double>, !cir.double
// CHECK-NEXT: %9 = cir.binop(mul, %7, %8) : !cir.double
