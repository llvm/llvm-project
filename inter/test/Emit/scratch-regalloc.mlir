// RUN: inter-opt %s --inter-regalloc --inter-insert-sync -o %t.mlir
// RUN: inter-translate %t.mlir --xemachine-to-ged -o %t.ged
// RUN: inter-ged-dump %t.ged | FileCheck %s --check-prefix=GED

module {
  func.func @scratch() attributes {
      xemachine.grf_count = 5 : i32,
      xemachine.kernel_type = () -> (),
      xemachine.reserved_grf_count = 0 : i32,
      xemachine.simd_size = 32 : i32,
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %root = xemachine.token
    %wide, %loaded = xemachine.load_slm %r0 dep %root {execSize = 32 : i32} : !xemachine.reg<16, 0> -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
    %extra, %loaded2 = xemachine.load_slm %r0 dep %root {execSize = 1 : i32} : !xemachine.reg<16, 0> -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %c1 = xemachine.imm 1 : i32
    %a = xemachine.mov %c1 {execSize = 1 : i32, noMask, xemachine.rematerialized} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %c2 = xemachine.imm 2 : i32
    %b = xemachine.mov %c2 {execSize = 1 : i32, noMask, xemachine.rematerialized} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %d = xemachine.add %a, %b {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %e = xemachine.add %d, %extra {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %f = xemachine.add %e, %wide {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<32, -1>, i32) -> !xemachine.reg<16, -1>
    return
  }
}

// GED: opcode=and exec=1
// GED: opcode=shr exec=1
// GED: opcode=send exec=1 {{.*}}sfid=ugm exdescRegFile=arf exdescAddrSubRegNum=2 exdescAddrSubRegRaw=4 desc=0x4200e504
// GED-SAME: len=2 eot=0
// GED: opcode=sync exec=1 {{.*}}function=allrd
// GED: opcode=send exec=1 {{.*}}sfid=ugm exdescRegFile=arf exdescAddrSubRegNum=2 exdescAddrSubRegRaw=4 desc=0x4220e500
