// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc},func.func(inter-insert-sync,inter-resource-info))' -o %t.mlir
// RUN: FileCheck %s --check-prefix=RESOURCE < %t.mlir
// RUN: inter-translate %t.mlir --xemachine-to-ged -o %t.ged
// RUN: inter-ged-dump %t.ged | FileCheck %s --check-prefix=GED

module {
  func.func @scratch() attributes {
      xemachine.grf_count = 128 : i32,
      xemachine.kernel_args = [],
      xemachine.reserved_grf_count = 123 : i32,
      xemachine.simd_size = 32 : i32,
      xemachine.slm_size = 128 : i64,
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
    %stored = xemachine.store_slm %r0 data %f dep %loaded2 {execSize = 1 : i32} : (!xemachine.reg<16, 0>, !xemachine.reg<16, -1>) -> !xemachine.mem.token
    %payload = xemachine.mov %r0 {noMask} : (!xemachine.reg<16, 0>, i32) -> !xemachine.reg<16, -1>
    xemachine.eot %payload dep %stored : !xemachine.reg<16, -1>
    return
  }
}

// GED: opcode=and exec=1
// GED: opcode=shr exec=1
// GED: opcode=send exec=1 {{.*}}sfid=ugm exdescRegFile=arf exdescAddrSubRegNum=2 exdescAddrSubRegRaw=4 desc=0x4200e504
// GED-SAME: len=2 eot=0
// GED: opcode=sync exec=1 {{.*}}function=allrd
// GED: opcode=send exec=1 {{.*}}sfid=ugm exdescRegFile=arf exdescAddrSubRegNum=2 exdescAddrSubRegRaw=4 desc=0x4220e500
// RESOURCE: func.func @scratch
// RESOURCE-SAME: xemachine.has_no_stateless_write = true
// RESOURCE: xemachine.send ugm {{.*}}xemachine.scratch_access
