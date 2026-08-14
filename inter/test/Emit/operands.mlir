// RUN: inter-opt %s --inter-insert-sync | inter-translate --xemachine-to-ged - -o %t
// RUN: inter-ged-dump %t | FileCheck %s

func.func @k() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  %one = xemachine.imm 1065353216 : f32
  // CHECK: pc=0 opcode=mov exec=16 swsb=0x0
  %f0 = xemachine.mov %one : (!xemachine.imm, f32) -> !xemachine.reg<16, 4>
  // CHECK-NEXT: pc=16 opcode=add exec=16 swsb=0x11
  %f1 = xemachine.add %f0, %f0 : (!xemachine.reg<16, 4>, !xemachine.reg<16, 4>, f32) -> !xemachine.reg<16, 5>

  %zero = xemachine.imm 0 : i32
  // CHECK-NEXT: pc=32 opcode=mov exec=16 swsb=0x0
  %i0 = xemachine.mov %zero : (!xemachine.imm, i32) -> !xemachine.reg<16, 6>
  // CHECK-NEXT: pc=48 opcode=add exec=16 swsb=0x9
  %mixed = xemachine.add %f1, %i0 : (!xemachine.reg<16, 5>, !xemachine.reg<16, 6>, i32) -> !xemachine.reg<16, 7>

  // CHECK-NEXT: pc=64 opcode=add exec=16 swsb=0x1a {{.*}}src0=grf6.0:ud:neg
  %sub = xemachine.sub %i0, %zero : (!xemachine.reg<16, 6>, !xemachine.imm, i32) -> !xemachine.reg<16, 8>
  // CHECK-NEXT: pc=80 opcode=or exec=16 swsb=0x19
  %or = xemachine.or %sub, %i0 : (!xemachine.reg<16, 8>, !xemachine.reg<16, 6>, i32) -> !xemachine.reg<16, 9>
  // CHECK-NEXT: pc=96 opcode=cmp exec=16 swsb=0x19 {{.*}}condition=gt flag=0.0
  %flag = xemachine.cmp gt %or, %zero : (!xemachine.reg<16, 9>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, 0>

  %bytes = xemachine.archreg 12 : !xemachine.reg<32, 12>
  // CHECK-NEXT: pc=112 opcode=mov exec=16 swsb=0x0 {{.*}}src0=grf12.16:ub
  %byte = xemachine.mov %bytes {src0Sub = 16 : i32, src0Type = i8} : (!xemachine.reg<32, 12>, i8) -> !xemachine.reg<16, 14>
  return
}
