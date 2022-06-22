// RUN: fir-opt %s | tco | FileCheck %s

// CHECK-LABEL: define [3 x i8] @x
func.func @x() -> !fir.char<1,3> {
  %1 = fir.string_lit "xyz"(3) : !fir.char<1,3>
 // CHECK: ret [3 x i8] c"xyz"
  return %1 : !fir.char<1,3>
}

// CHECK-LABEL: define x86_fp80 @y()
func.func @y() -> !fir.real<10> {
 %c1 = arith.constant 42.4 :f32
 %0 = fir.convert %c1 : (f32) -> !fir.real<10>
  // CHECK: ret x86_fp80 0xK4004A9999A0000000000
  // TODO: What's that number?
  return %0 : !fir.real<10>
}

// CHECK-LABEL: define i16 @z()
func.func @z() -> !fir.logical<2> {
  %1 = arith.constant true
  %0 = fir.convert %1 : (i1) -> !fir.logical<2>
 // CHECK-LABEL: ret i16 1
  return %0 : !fir.logical<2>
}
