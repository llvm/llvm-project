// RUN: not inter-translate --xemachine-to-zebin %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: required work-group size must contain three positive integers

module {
  func.func @kernel() attributes {
      xemachine.kernel_args = [],
      xemachine.required_work_group_size = [1 : i32],
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
    %token = xemachine.token
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    xemachine.eot %r0 dep %token : !xemachine.reg<16, 0>
    return
  }
}
