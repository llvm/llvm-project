// RUN: inter-opt %s --split-input-file --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' | FileCheck %s --check-prefixes=REMAT,SCRATCH,LOOP

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
  func.func @scratch() attributes {xemachine.grf_count = 6 : i32, xemachine.reserved_grf_count = 0 : i32, xemachine.scratch_size = 128 : i64} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %root = xemachine.token
    %zero = xemachine.imm 0 : i32
    %address = xemachine.mov %zero {execSize = 1 : i32, noMask, xemachine.rematerialized} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %wide, %loaded = xemachine.load_slm %address dep %root {execSize = 32 : i32} : !xemachine.reg<16, -1> -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
    %extra, %loaded2 = xemachine.load_slm %address dep %root {execSize = 1 : i32} : !xemachine.reg<16, -1> -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %c1 = xemachine.imm 1 : i32
    %a = xemachine.mov %c1 {execSize = 1 : i32, noMask, xemachine.rematerialized} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %c2 = xemachine.imm 2 : i32
    %b = xemachine.mov %c2 {execSize = 1 : i32, noMask, xemachine.rematerialized} : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %d = xemachine.add %a, %b {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %e = xemachine.add %d, %extra {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<16, -1>, i32) -> !xemachine.reg<16, -1>
    %f = xemachine.add %e, %wide {execSize = 1 : i32, noMask} : (!xemachine.reg<16, -1>, !xemachine.reg<32, -1>, i32) -> !xemachine.reg<16, -1>
    %joined = xemachine.token_join %loaded, %loaded2 : !xemachine.mem.token, !xemachine.mem.token
    %stored = xemachine.store_slm %r0 data %f dep %joined {execSize = 1 : i32} : (!xemachine.reg<16, 0>, !xemachine.reg<16, -1>) -> !xemachine.mem.token
    return
  }
}

// SCRATCH-LABEL: func.func @scratch
// SCRATCH-SAME: xemachine.regalloc_iterations = 2
// SCRATCH-SAME: xemachine.scratch_size = 256
// SCRATCH: xemachine.shr {{.*}}xemachine.scratch_setup
// SCRATCH: xemachine.send ugm {{.*}} data {{.*}} exdesc
// SCRATCH: xemachine.load_slm
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

// -----

module {
  func.func @allocate_flags() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %zero = xemachine.imm 0 : i32
    %first = xemachine.cmp eq %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    %second = xemachine.cmp ne %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    xemachine.uniform_if %first : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    xemachine.uniform_if %second : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    %reused = xemachine.cmp gt %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    xemachine.uniform_if %reused : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    return
  }
}

// LOOP-LABEL: func.func @allocate_flags
// LOOP: %[[FIRST:.*]] = xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 0>
// LOOP: %[[SECOND:.*]] = xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 1>
// LOOP: xemachine.uniform_if %[[FIRST]] : !xemachine.arf<f, 2, 0>
// LOOP: xemachine.uniform_if %[[SECOND]] : !xemachine.arf<f, 2, 1>
// LOOP: xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 0>
// LOOP-NOT: !xemachine.arf<f, 2, -1>

// -----

module {
  func.func @reserve_fixed_flag() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %zero = xemachine.imm 0 : i32
    %fixed = xemachine.cmp eq %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, 0>
    %allocated = xemachine.cmp ne %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    xemachine.uniform_if %fixed : !xemachine.arf<f, 2, 0> {
      xemachine.yield
    }
    xemachine.uniform_if %allocated : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    return
  }
}

// LOOP-LABEL: func.func @reserve_fixed_flag
// LOOP: xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 0>
// LOOP: xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 1>
// LOOP-NOT: !xemachine.arf<f, 2, -1>

// -----

module {
  func.func @loop_flag_liveness() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %zero = xemachine.imm 0 : i32
    %captured = xemachine.cmp eq %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    xemachine.uniform_loop () {
      xemachine.uniform_if %captured : !xemachine.arf<f, 2, -1> {
        xemachine.yield
      }
      %inside = xemachine.cmp ne %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
      xemachine.continue_if %inside : !xemachine.arf<f, 2, -1>
    } : () -> ()
    return
  }
}

// LOOP-LABEL: func.func @loop_flag_liveness
// LOOP: %[[CAPTURED:.*]] = xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 0>
// LOOP: xemachine.uniform_loop
// LOOP: xemachine.uniform_if %[[CAPTURED]] : !xemachine.arf<f, 2, 0>
// LOOP: xemachine.cmp {{.*}} -> !xemachine.arf<f, 2, 1>
// LOOP-NOT: !xemachine.arf<f, 2, -1>
