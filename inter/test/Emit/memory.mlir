// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

module {
  func.func @memory() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
    %root = xemachine.token
    %addr = xemachine.archreg 4 : !xemachine.reg<32, 4>
    %data = xemachine.archreg 8 : !xemachine.reg<32, 8>
    %a64 = xemachine.archreg 20 : !xemachine.reg<64, 20>

    // CHECK: opcode=send exec=32 {{.*}}sfid=slm exdescRegFile=imm exdesc=0x0 desc=0x4200500 {{.*}}dst=grf12 src0=grf4 src1=arf0 len=0 eot=0
    %loaded, %load = xemachine.load_slm %addr dep %root : !xemachine.reg<32, 4> -> (!xemachine.reg<32, 12>, !xemachine.mem.token)
    // CHECK-NEXT: {{.*}}opcode=send exec=32 {{.*}}sfid=slm exdescRegFile=imm exdesc=0x0 desc=0x4000504 {{.*}}dst=arf0 src0=grf4 src1=grf8 len=2 eot=0
    %store = xemachine.store_slm %addr data %data dep %load : (!xemachine.reg<32, 4>, !xemachine.reg<32, 8>) -> !xemachine.mem.token
    // CHECK-NEXT: {{.*}}opcode=send exec=32 {{.*}}sfid=ugm exdescRegFile=imm exdesc=0x0 desc=0x820058c {{.*}}dst=grf14 src0=grf20 src1=grf8 len=2 eot=0
    %old, %atomic = xemachine.atomic_iadd_a64 %a64 data %data dep %store : (!xemachine.reg<64, 20>, !xemachine.reg<32, 8>) -> (!xemachine.reg<32, 14>, !xemachine.mem.token)

    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    // CHECK-NEXT: {{.*}}opcode=send exec=1 {{.*}}sfid=slm exdescRegFile=imm exdesc=0x0 desc=0x210001f {{.*}}dst=grf16 src0=grf0 src1=arf0 len=0 eot=0
    %readback, %fence = xemachine.fence_slm %r0 dep %atomic : !xemachine.reg<16, 0> -> (!xemachine.reg<16, 16>, !xemachine.mem.token)
    // CHECK-NEXT: {{.*}}opcode=mov exec=8 {{.*}}dst=arf0.0:ud<1> src0=grf16.0:ud<1;1,0>
    %await = xemachine.fence_await %readback dep %fence : !xemachine.reg<16, 16> -> !xemachine.mem.token
    // CHECK-NEXT: {{.*}}opcode=send exec=1 {{.*}}sfid=gateway exdescRegFile=imm exdesc=0x0 desc=0x2000004 {{.*}}dst=arf0 src0=grf8 src1=arf0 len=0 eot=0
    %signal = xemachine.barrier_signal %data dep %await : !xemachine.reg<32, 8> -> !xemachine.mem.token
    // CHECK-NEXT: {{.*}}opcode=sync exec=1 {{.*}}function=bar
    %bar = xemachine.sync bar dep %signal : !xemachine.mem.token
    // CHECK-NEXT: {{.*}}opcode=send exec=1 {{.*}}sfid=gateway exdescRegFile=imm exdesc=0x0 desc=0x2000010 {{.*}}dst=arf0 src0=grf8 src1=arf0 len=0 eot=1
    xemachine.eot %data dep %bar : !xemachine.reg<32, 8>
    return
  }
}
