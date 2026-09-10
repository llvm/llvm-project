// RUN: inter-opt %s --inter-resource-info --inter-insert-sync -o %t.mlir
// RUN: inter-translate %t.mlir --xemachine-to-ged -o %t
// RUN: inter-ged-dump %t | FileCheck %s
// RUN: inter-translate %t.mlir --xemachine-to-zebin -o %t.zebin
// RUN: llvm-readobj --string-dump=.ze_info %t.zebin | FileCheck %s --check-prefix=ZEINFO

module {
  func.func @scratch_exdesc() attributes {
      xemachine.grf_count = 128 : i32,
      xemachine.kernel_args = [],
      xemachine.reserved_grf_count = 5 : i32,
      xemachine.scratch_size = 128 : i64,
      xemachine.simd_size = 32 : i32,
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %mask = xemachine.imm 4294966272 : i32
    %masked = xemachine.and %r0, %mask {dstRegion = #xemachine.dstregion<1>, dstSub = 2 : i32, execSize = 1 : i32, noMask, src0Region = #xemachine.region<0, 1, 0>, src0Sub = 5 : i32} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<a0, 16, 0>
    %four = xemachine.imm 4 : i32
    %sso = xemachine.shr %masked, %four {dstRegion = #xemachine.dstregion<1>, dstSub = 2 : i32, execSize = 1 : i32, noMask, src0Region = #xemachine.region<0, 1, 0>, src0Sub = 2 : i32} : (!xemachine.arf<a0, 16, 0>, !xemachine.imm, i32) -> !xemachine.arf<a0, 16, 0>
    %zero = xemachine.imm 0 : i32
    %address = xemachine.mov %zero {dstRegion = #xemachine.dstregion<1>, execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, 2>
    %root = xemachine.token
    %dst, %token = xemachine.send ugm %address exdesc %sso : !xemachine.arf<a0, 16, 0> dep %root {desc = 1109452032 : i32, exdesc = 0 : i32, execSize = 1 : i32, noMask, sfid = 0 : i32} : (!xemachine.reg<16, 2>) -> (!xemachine.reg<32, 4>, !xemachine.mem.token)
    %payload = xemachine.mov %r0 {dstRegion = #xemachine.dstregion<1>, noMask, src0Region = #xemachine.region<1, 1, 0>} : (!xemachine.reg<16, 0>, i32) -> !xemachine.reg<16, 6>
    xemachine.eot %payload dep %token : !xemachine.reg<16, 6>
    return
  }
}

// CHECK: opcode=and exec=1 swsb=0x11
// CHECK: opcode=shr exec=1 swsb=0x19
// CHECK: opcode=send exec=1
// CHECK-SAME: sfid=ugm
// CHECK-SAME: exdescRegFile=arf
// CHECK-SAME: exdescAddrSubRegNum=2
// CHECK-SAME: exdescAddrSubRegRaw=4
// ZEINFO: has_no_stateless_write: true
// ZEINFO: spill_size: 128
// ZEINFO: per_thread_memory_buffers:
// ZEINFO: type: scratch
// ZEINFO: usage: single_space
// ZEINFO: size: 128
