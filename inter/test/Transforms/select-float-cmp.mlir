// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @float_cmp() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 16 : i32} {
    %one = xw.constant 1.0 : f32 -> !xw.simd<f32, 16>
    %une = xw.cmpf une %one, %one
        : !xw.simd<f32, 16>, !xw.simd<f32, 16> -> !xw.mask<16>
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @float_cmp
// CHECK: xemachine.cmp ne
