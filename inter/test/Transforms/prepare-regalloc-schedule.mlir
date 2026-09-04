// RUN: inter-opt %s --pass-pipeline='builtin.module(func.func(inter-prepare-regalloc,inter-machine-schedule))' | FileCheck %s

module {
  func.func @repair_before_schedule() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %value = xemachine.mov %one {execSize = 16 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %tuple = xemachine.tuple_from_elements %value, %value
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<32, -1>
    %producer = xemachine.add %r0, %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, f32)
        -> !xemachine.reg<16, -1>
    %consumer = xemachine.add %producer, %producer {execSize = 16 : i32}
        : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, f32)
        -> !xemachine.reg<16, -1>
    return
  }
}

// CHECK-LABEL: func.func @repair_before_schedule
// CHECK: [[VALUE:%.*]] = xemachine.mov
// CHECK-NEXT: [[COPY:%.*]] = xemachine.mov [[VALUE]] {{.*}}xemachine.regalloc_copy = "tuple-element"
// CHECK-NEXT: xemachine.tuple_from_elements [[VALUE]], [[COPY]]
