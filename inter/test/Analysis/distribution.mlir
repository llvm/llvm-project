// RUN: inter-opt %s --inter-refine-distribution='simd-width=16' | FileCheck %s

func.func @distribution(%uniform: i32)
    attributes {xw.simd_width = 16 : i64} {
  %constant = xw.constant 1 : i32
  %gid = xw.global_id 0 : !xw.simd<i32, 16>
  %sum = xw.binary addi %gid, %constant
      : !xw.simd<i32, 16>, i32 -> !xw.simd<i32, 16>
  return
}

// CHECK-LABEL: func.func @distribution
// CHECK-SAME: xw.distribution = 1 : i32
// CHECK: xw.constant {{.*}}xw.distribution = array<i32: 1>
// CHECK: xw.global_id {{.*}}xw.distribution = array<i32: 16>
// CHECK: xw.binary {{.*}}xw.distribution = array<i32: 16>
// CHECK-NOT: llvm.
// CHECK-NOT: cf.
