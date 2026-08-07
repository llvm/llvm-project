// Emitter text forms for ALU ops, sends, and sync.
// RUN: inter-translate --xemachine-to-iga %s | FileCheck %s

func.func @k() {
  %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
  %c = xemachine.imm 4294967232 : i32
  // CHECK: (W)     and (1|M0)  r4.0<1>:ud  r0.0<0;1,0>:ud  0xffffffc0:ud
  %base = xemachine.and %r0, %c {execSize = 1 : i32, noMask, src0Region = #xemachine.region<0, 1, 0>} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.reg<16, 4>
  // CHECK: add (32|M0)  r6.0<1>:ud  r4.0<1;1,0>:ud  r4.0<1;1,0>:ud  {I@1}
  %sum = xemachine.add %base, %base {execSize = 32 : i32} : (!xemachine.reg<16, 4>, !xemachine.reg<16, 4>, i32) -> !xemachine.reg<32, 6>
  // CHECK: (W)     send.ugm (1|M0)  r8.0  r4.0  null:0  0xff000000  0x6219d500{{ *}}{A@2,$0}
  %dst, %tok = xemachine.send ugm %base {desc = 1645860096 : i32, exdesc = -16777216 : i32, noMask, sfid = 0 : i32} : (!xemachine.reg<16, 4>) -> (!xemachine.reg<16, 8>, !xemachine.mem.token)
  // CHECK: sync.allrd null
  xemachine.sync allrd
  // CHECK: send.ugm (32|M0)  null  r4.0  r6.0:2  0x0  0x8000584{{ *}}{A@3,$1}
  %n, %tok2 = xemachine.send ugm %base data %sum {desc = 134219140 : i32, exdesc = 0 : i32, execSize = 32 : i32, sfid = 0 : i32} : (!xemachine.reg<16, 4>, !xemachine.reg<32, 6>) -> (!xemachine.reg<0, 9>, !xemachine.mem.token)
  return
}
