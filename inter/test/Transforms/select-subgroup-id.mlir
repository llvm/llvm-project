// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @subgroup_id() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 16 : i32} {
    %id = xw.subgroup_id : i32
    return
  }

  func.func @group_ids() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 16 : i32} {
    %x = xw.group_id 0 : i64
    %y = xw.group_id 1 : i64
    %z = xw.group_id 2 : i64
    return
  }
}

// CHECK-LABEL: func.func @subgroup_id
// CHECK-NOT: xemachine.payload_prologue
// CHECK: xemachine.and {{.*}}src0Sub = 2 : i32

// CHECK-LABEL: func.func @group_ids
// CHECK-DAG: xemachine.mov {{.*}}src0Sub = 1 : i32{{.*}}src0Type = i32
// CHECK-DAG: xemachine.mov {{.*}}src0Sub = 6 : i32{{.*}}src0Type = i32
// CHECK-DAG: xemachine.mov {{.*}}src0Sub = 7 : i32{{.*}}src0Type = i32
