// RUN: not inter-opt %s --inter-regalloc 2>&1 | FileCheck %s

module {
  func.func @fixed_conflict() attributes {xemachine.grf_count = 2 : i32, xemachine.reserved_grf_count = 0 : i32} {
    %c1 = xemachine.imm 1 : i32
    %a = xemachine.mov %c1 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, 0>
    %c2 = xemachine.imm 2 : i32
    %b = xemachine.mov %c2 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, 0>
    %sum = xemachine.add %a, %b {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, i32) -> !xemachine.reg<16, -1>
    return
  }
}

// CHECK: register allocation exhausted 2 GRFs
