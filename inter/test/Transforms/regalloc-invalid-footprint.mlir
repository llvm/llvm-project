// RUN: not inter-opt %s --inter-regalloc 2>&1 | FileCheck %s

module {
  func.func @invalid_footprint() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %wide = xemachine.mov %r0 {execSize = 32 : i32, noMask, src0Region = #xemachine.region<1, 1, 0>} : (!xemachine.reg<16, 0>, i32) -> !xemachine.reg<32, -1>
    return
  }
}

// CHECK: source region exceeds declared register storage
