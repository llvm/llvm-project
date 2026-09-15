// GED encoding for ALU ops, sends, and sync.
// RUN: inter-opt %s --inter-insert-sync | inter-translate --xemachine-to-ged - -o %t
// RUN: inter-ged-dump %t | FileCheck %s
// RUN: inter-opt %s --inter-insert-sync | inter-translate --xemachine-to-asm - | FileCheck %s --check-prefix=ASM

func.func @k() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  %root = xemachine.token
  %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
  %c = xemachine.imm 4294967232 : i32
  // CHECK: pc=0 opcode=and exec=1 swsb=0x0 mask=nomask channel=0 pred=normal dst=grf4.0:ud<1> src0=grf0.0:ud<0;1,0> src1=imm0xffffffc0:ud
  %base = xemachine.and %r0, %c {execSize = 1 : i32, noMask, src0Region = #xemachine.region<0, 1, 0>} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.reg<16, 4>
  // CHECK-NEXT: pc=16 opcode=add exec=32 swsb=0x19 mask=normal channel=0 pred=normal dst=grf6.0:ud<1> src0=grf4.0:ud<1;1,0> src1=grf4.0:ud<1;1,0>
  %sum = xemachine.add %base, %base {execSize = 32 : i32} : (!xemachine.reg<16, 4>, !xemachine.reg<16, 4>, i32) -> !xemachine.reg<32, 6>
  // CHECK-NEXT: pc=32 opcode=send exec=1 swsb=0x340 sfid=ugm exdescRegFile=imm exdesc=0xff000000 desc=0x6219d500 mask=nomask channel=0 pred=normal dst=grf8 src0=grf4 src1=arf0 len=0 eot=0
  %dst, %tok = xemachine.send ugm %base dep %root {desc = 1645860096 : i32, exdesc = -16777216 : i32, noMask, sfid = 0 : i32} : (!xemachine.reg<16, 4>) -> (!xemachine.reg<16, 8>, !xemachine.mem.token)
  %after = xemachine.after %tok : !xemachine.mem.token
  %joined = xemachine.token_join %root, %after : !xemachine.mem.token, !xemachine.mem.token
  // CHECK-NEXT: pc=48 opcode=sync exec=1 swsb=0x0 function=allrd mask=normal channel=0 pred=normal
  %t3 = xemachine.sync allrd dep %joined : !xemachine.mem.token
  // CHECK-NEXT: pc=64 opcode=send exec=32 swsb=0x321 sfid=ugm exdescRegFile=imm exdesc=0x0 desc=0x8000584 mask=normal channel=0 pred=normal dst=arf0 src0=grf4 src1=grf6 len=2 eot=0
  %n, %tok2 = xemachine.send ugm %base data %sum dep %t3 {desc = 134219140 : i32, exdesc = 0 : i32, execSize = 32 : i32, sfid = 0 : i32} : (!xemachine.reg<16, 4>, !xemachine.reg<32, 6>) -> (!xemachine.reg<0, 9>, !xemachine.mem.token)
  %eot_payload = xemachine.archreg 0 : !xemachine.reg<16, 10>
  // CHECK-NEXT: pc=80 opcode=sync exec=1 swsb=0x80 function=nop
  // CHECK-NEXT: pc=96 opcode=sync exec=1 swsb=0xa1 function=nop
  // CHECK-NEXT: pc=112 opcode=send exec=1 swsb=0xc2 sfid=gateway exdescRegFile=imm exdesc=0x0 desc=0x2000010 mask=nomask channel=0 pred=normal dst=arf0 src0=grf10 src1=arf0 len=0 eot=1
  xemachine.eot %eot_payload dep %tok2 : !xemachine.reg<16, 10>
  return
}

// ASM: L0:
// ASM-NEXT: and (1|M0)
// ASM-SAME: r4.0<1>:ud
// ASM-SAME: r0.0<0;1,0>:ud
// ASM-NEXT: add (32|M0)
// ASM-SAME: r6.0<1>:ud
// ASM-SAME: {I@1}
// ASM-NEXT: send.ugm (1|M0)
// ASM-SAME: load.ugm.d32x16t.a32.ca.cc.bti[255]
// ASM-NEXT: sync.allrd
// ASM-NEXT: send.ugm (32|M0)
// ASM-SAME: store.ugm.d32.a64
// ASM-NEXT: sync.nop
// ASM-SAME: {$0.dst}
// ASM-NEXT: sync.nop
// ASM-SAME: {$1.src}
// ASM-NEXT: send.gtwy (1|M0)
// ASM-SAME: {EOT,$2}
