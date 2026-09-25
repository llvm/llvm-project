// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @local_i32_offset() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 16 : i32} {
    %base = xw.local_memory_base : !xw.ptr<#xw.local>
    %offset = xw.constant 31 : i32 -> !xw.simd<i32, 16>
    %address = xw.ptradd %base, %offset : !xw.ptr<#xw.local>, !xw.simd<i32, 16> -> !xw.simd<!xw.ptr<#xw.local>, 16>
    return
  }

  func.func @barrier_header() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 16 : i32} {
    %lid = xw.local_id 0 : !xw.simd<i64, 16>
    %root = xw.token : !xw.mem.token
    %barrier = xw.barrier %root : !xw.mem.token -> !xw.mem.token
    return
  }
}

// CHECK-LABEL: func.func @local_i32_offset
// CHECK: xemachine.add
// CHECK-LABEL: func.func @barrier_header
// CHECK: %[[INLINE:.*]] = xemachine.archreg 2
// CHECK: xemachine.fence_slm %[[INLINE]]
// CHECK: xemachine.mov %[[INLINE]] {{.*}}src0Sub = 11
