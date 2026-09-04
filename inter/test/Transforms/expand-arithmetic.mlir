// RUN: inter-opt %s --inter-expand-arithmetic | FileCheck %s

// CHECK-LABEL: func.func @unsigned
// CHECK-NOT: xw.binary divui
// CHECK-NOT: xw.binary remui
// CHECK: xw.binary shrui
// CHECK: xw.cmpi uge
// CHECK: xw.select
func.func @unsigned(%lhs: i8, %rhs: i8) -> (i8, i8) {
  %div = xw.binary divui %lhs, %rhs : i8, i8 -> i8
  %rem = xw.binary remui %lhs, %rhs : i8, i8 -> i8
  return %div, %rem : i8, i8
}

// CHECK-LABEL: func.func @signed
// CHECK-NOT: xw.binary divsi
// CHECK-NOT: xw.binary remsi
// CHECK: xw.cmpi slt
// CHECK: xw.binary xori
// CHECK: xw.select
func.func @signed(%lhs: i8, %rhs: i8) -> (i8, i8) {
  %div = xw.binary divsi %lhs, %rhs : i8, i8 -> i8
  %rem = xw.binary remsi %lhs, %rhs : i8, i8 -> i8
  return %div, %rem : i8, i8
}

// CHECK-LABEL: func.func @simd_signed
// CHECK: xw.mask_xor
func.func @simd_signed(%lhs: !xw.simd<i8, 16>, %rhs: !xw.simd<i8, 16>)
    -> !xw.simd<i8, 16> attributes {xw.simd_width = 16 : i32} {
  %div = xw.binary divsi %lhs, %rhs
      : !xw.simd<i8, 16>, !xw.simd<i8, 16> -> !xw.simd<i8, 16>
  return %div : !xw.simd<i8, 16>
}
