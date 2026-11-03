// RUN: inter-opt %s --inter-prepare-regalloc | FileCheck %s

module {
  func.func @wide_immediate() {
    %value = xemachine.archreg 4 : !xemachine.reg<16, 4>
    %one = xemachine.imm 1 : i64
    %result = xemachine.and %value, %one {execSize = 1 : i32, noMask,
        src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<16, 4>, !xemachine.imm, i64)
        -> !xemachine.reg<2, -1>
    return
  }
}

// CHECK: %[[ONE:.*]] = xemachine.imm 1 : i64
// CHECK: %[[MATERIALIZED:.*]] = xemachine.mov %[[ONE]] {{.*}}xemachine.immediate_legalization
// CHECK: xemachine.and {{.*}}, %[[MATERIALIZED]]
// CHECK-SAME: src1Region = #xemachine.region<0, 1, 0>
