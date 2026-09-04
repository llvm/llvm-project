// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @subgroup_id() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.required_work_group_size = [256 : i32, 1 : i32, 1 : i32],
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
// CHECK-SAME: xemachine.per_thread_payload_size = 64 : i32
// CHECK: xemachine.payload_prologue
// CHECK: xemachine.and {{.*}}src0Sub = 4 : i32{{.*}}, i16)
// CHECK: xemachine.shr

// CHECK-LABEL: func.func @group_ids
// CHECK-DAG: xemachine.mov {{.*}}src0Sub = 1 : i32{{.*}}src0Type = i32
// CHECK-DAG: xemachine.mov {{.*}}src0Sub = 6 : i32{{.*}}src0Type = i32
// CHECK-DAG: xemachine.mov {{.*}}src0Sub = 7 : i32{{.*}}src0Type = i32
