// RUN: fir-opt %s | tco | FileCheck %s

// CHECK-LABEL: define [3 x i8] @x
func.func @x() -> !fir.char<1,3> {
  %1 = fir.string_lit "xyz"(3) : !fir.char<1,3>
 // CHECK: ret [3 x i8] c"xyz"
  return %1 : !fir.char<1,3>
}

// CHECK-LABEL: define i16 @z()
func.func @z() -> !fir.logical<2> {
  %1 = arith.constant true
  %0 = fir.convert %1 : (i1) -> !fir.logical<2>
 // CHECK-LABEL: ret i16 1
  return %0 : !fir.logical<2>
}
