// RUN: inter-opt %s --split-input-file --inter-regalloc | FileCheck %s --check-prefixes=REMAT,SCRATCH,LOOP

module {
  func.func @rematerialize() attributes {xemachine.grf_count = 5 : i32, xemachine.reserved_grf_count = 0 : i32} {
    %c1 = xemachine.imm 1 : i32
    %a = xemachine.mov %c1 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %c2 = xemachine.imm 2 : i32
    %b = xemachine.mov %c2 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %c3 = xemachine.imm 3 : i32
    %c = xemachine.mov %c3 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %c4 = xemachine.imm 4 : i32
    %f = xemachine.mov %c4 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %c5 = xemachine.imm 5 : i32
    %h = xemachine.mov %c5 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %d = xemachine.add %a, %b {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %e = xemachine.add %d, %c {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %g = xemachine.add %e, %f {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %i = xemachine.add %g, %h {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    return
  }
}

// REMAT-LABEL: func.func @rematerialize
// REMAT-SAME: xemachine.regalloc_iterations = 2
// REMAT: xemachine.mov {{.*}}xemachine.rematerialized
// REMAT: xemachine.mov {{.*}}xemachine.rematerialized
// REMAT-NOT: !xemachine.reg<{{.*}}, -1>

// -----

module {
  func.func @scratch() attributes {xemachine.grf_count = 5 : i32, xemachine.reserved_grf_count = 0 : i32, xemachine.scratch_size = 128 : i64} {
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

// SCRATCH-LABEL: func.func @scratch
// SCRATCH-SAME: xemachine.regalloc_iterations = 2
// SCRATCH-SAME: xemachine.scratch_size = 256
// SCRATCH: xemachine.shr {{.*}}xemachine.scratch_setup
// SCRATCH: xemachine.load_slm
// SCRATCH: xemachine.send ugm {{.*}} data {{.*}} exdesc
// SCRATCH: xemachine.send ugm {{.*}} exdesc {{.*}} dep
// SCRATCH-NOT: !xemachine.reg<{{.*}}, -1>

// -----

module {
  func.func @loop_capture() attributes {xemachine.grf_count = 2 : i32, xemachine.reserved_grf_count = 0 : i32} {
    %c1 = xemachine.imm 1 : i32
    %outside = xemachine.mov %c1 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %flag = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
    xemachine.uniform_loop () {
      %c2 = xemachine.imm 2 : i32
      %use = xemachine.add %outside, %c2 {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.imm, i32) -> !xemachine.reg<16, -1>
      %c3 = xemachine.imm 3 : i32
      %later = xemachine.mov %c3 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
      xemachine.continue_if %flag : !xemachine.arf<f, 2, 0>
    } : () -> ()
    return
  }
}

// LOOP-LABEL: func.func @loop_capture
// LOOP: [[OUTSIDE:%.*]] = xemachine.mov {{.*}}-> !xemachine.reg<16, 0>
// LOOP: xemachine.uniform_loop
// LOOP: xemachine.add [[OUTSIDE]], {{.*}}-> !xemachine.reg<16, 1>
// LOOP: xemachine.mov {{.*}}-> !xemachine.reg<16, 1>
