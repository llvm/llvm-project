// RUN: not inter-opt --inter-select-to-machine %s 2>&1 | FileCheck %s

// CHECK: xw.required_work_group_size must contain three positive integers

module {
  func.func @kernel() attributes {
      xw.kernel,
      xw.kernel_args = [],
      xw.required_work_group_size = [1 : i32],
      xw.simd_width = 8 : i32} {
    %id = xw.subgroup_id : i32
    return
  }
}
