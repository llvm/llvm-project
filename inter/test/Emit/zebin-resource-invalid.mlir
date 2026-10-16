// RUN: not inter-translate --xemachine-to-zebin %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: SLM and scratch sizes must be nonnegative

module {
  func.func @kernel() attributes {
      xemachine.kernel_args = [],
      xemachine.slm_size = -1 : i64,
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
    %token = xemachine.token
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    xemachine.eot %r0 dep %token : !xemachine.reg<16, 0>
    return
  }
}
