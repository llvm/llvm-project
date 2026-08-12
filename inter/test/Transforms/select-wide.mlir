// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @wide_address(%base: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 32 : i32} {
    %gid = xw.global_id 0 : !xw.simd<i64, 32>
    %four = xw.constant 4 : i64 -> !xw.simd<i64, 32>
    %offset = xw.binary addi %gid, %four : !xw.simd<i64, 32>, !xw.simd<i64, 32> -> !xw.simd<i64, 32>
    %address = xw.ptradd %base, %offset : !xw.ptr<#xw.global>, !xw.simd<i64, 32> -> !xw.simd<!xw.ptr<#xw.global>, 32>
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @wide_address
// SIMD32 i64 is decomposed into two SIMD16 machine operations.
// CHECK-COUNT-4: xemachine.add
// CHECK-NOT: xw.wide
