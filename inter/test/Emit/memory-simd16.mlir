// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

module {
  func.func @memory_simd16() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
    %root = xemachine.token
    %addr = xemachine.archreg 4 : !xemachine.reg<16, 4>
    %data = xemachine.archreg 8 : !xemachine.reg<16, 8>
    %a64 = xemachine.archreg 20 : !xemachine.reg<32, 20>

    // CHECK: opcode=send exec=16 {{.*}}sfid=slm exdescRegFile=imm exdesc=0x0 desc=0x2100500 {{.*}}dst=grf12 src0=grf4 src1=arf0 len=0 eot=0
    %loaded, %load = xemachine.load_slm %addr dep %root {execSize = 16 : i32} : !xemachine.reg<16, 4> -> (!xemachine.reg<16, 12>, !xemachine.mem.token)
    // CHECK-NEXT: {{.*}}opcode=send exec=16 {{.*}}sfid=slm exdescRegFile=imm exdesc=0x0 desc=0x2000504 {{.*}}dst=arf0 src0=grf4 src1=grf8 len=1 eot=0
    %store = xemachine.store_slm %addr data %data dep %load {execSize = 16 : i32} : (!xemachine.reg<16, 4>, !xemachine.reg<16, 8>) -> !xemachine.mem.token
    // CHECK-NEXT: {{.*}}opcode=send exec=16 {{.*}}sfid=ugm exdescRegFile=imm exdesc=0x0 desc=0x410058c {{.*}}dst=grf14 src0=grf20 src1=grf8 len=1 eot=0
    %old, %atomic = xemachine.atomic_iadd_a64 %a64 data %data dep %store {execSize = 16 : i32} : (!xemachine.reg<32, 20>, !xemachine.reg<16, 8>) -> (!xemachine.reg<16, 14>, !xemachine.mem.token)
    // CHECK-NEXT: {{.*}}opcode=send exec=16 {{.*}}sfid=ugm exdescRegFile=imm exdesc=0x0 desc=0x4100580 {{.*}}dst=grf16 src0=grf20 src1=arf0 len=0 eot=0
    %global_loaded, %global_load = xemachine.load_a64 %a64 dep %atomic {execSize = 16 : i32} : !xemachine.reg<32, 20> -> (!xemachine.reg<16, 16>, !xemachine.mem.token)
    // CHECK-NEXT: {{.*}}opcode=send exec=16 {{.*}}sfid=ugm exdescRegFile=imm exdesc=0x0 desc=0x4000584 {{.*}}dst=arf0 src0=grf20 src1=grf16 len=1 eot=0
    %global_store = xemachine.store_a64 %a64 data %global_loaded dep %global_load {execSize = 16 : i32} : (!xemachine.reg<32, 20>, !xemachine.reg<16, 16>) -> !xemachine.mem.token
    return
  }
}
