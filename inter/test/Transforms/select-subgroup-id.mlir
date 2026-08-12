// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @subgroup_id() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 16 : i32} {
    %id = xw.subgroup_id : i32
    return
  }
}

// CHECK-LABEL: func.func @subgroup_id
// CHECK-NOT: xemachine.load_block_a32
// CHECK: %[[R0:.*]] = xemachine.archreg 0
// CHECK: xemachine.and %[[R0]], {{.*}} {{.*}}execSize = 1 : i32{{.*}}noMask{{.*}}src0Region = #xemachine.region<0, 1, 0>{{.*}}src0Sub = 2 : i32
