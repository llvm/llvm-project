// RUN: not inter-opt %s --split-input-file --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' 2>&1 | FileCheck %s

module {
  func.func @fixed_conflict() attributes {xemachine.grf_count = 2 : i32, xemachine.reserved_grf_count = 0 : i32} {
    %c1 = xemachine.imm 1 : i32
    %a = xemachine.mov %c1 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, 0>
    %c2 = xemachine.imm 2 : i32
    %b = xemachine.mov %c2 {execSize = 1 : i32, noMask} : (!xemachine.imm, i32) -> !xemachine.reg<16, 0>
    %sum = xemachine.add %a, %b {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, i32) -> !xemachine.reg<16, -1>
    return
  }
}

// CHECK: register allocation exhausted 2 GRFs

// -----

module {
  func.func @flag_pressure() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %zero = xemachine.imm 0 : i32
    %first = xemachine.cmp eq %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    %second = xemachine.cmp ne %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    %third = xemachine.cmp gt %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, -1>
    xemachine.uniform_if %first : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    xemachine.uniform_if %second : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    xemachine.uniform_if %third : !xemachine.arf<f, 2, -1> {
      xemachine.yield
    }
    return
  }
}

// CHECK: flag allocation exhausted f0/f1 for overlapping live ranges

// -----

module {
  func.func @fixed_flag_conflict() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %zero = xemachine.imm 0 : i32
    %first = xemachine.cmp eq %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, 0>
    %second = xemachine.cmp ne %r0, %zero {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<f, 2, 0>
    xemachine.uniform_if %first : !xemachine.arf<f, 2, 0> {
      xemachine.yield
    }
    xemachine.uniform_if %second : !xemachine.arf<f, 2, 0> {
      xemachine.yield
    }
    return
  }
}

// CHECK: fixed f0 live ranges overlap

// -----

module {
  func.func @unsupported_virtual_acc() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %one = xemachine.imm 1 : i32
    %acc = xemachine.mul %r0, %one {execSize = 1 : i32, noMask} : (!xemachine.reg<16, 0>, !xemachine.imm, i32) -> !xemachine.arf<acc, 16, -1>
    return
  }
}

// CHECK: virtual acc ARF allocation is unsupported
