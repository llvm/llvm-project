// RUN: not inter-opt %s --split-input-file --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' 2>&1 | FileCheck %s

module {
  func.func @invalid_footprint() attributes {xemachine.grf_count = 4 : i32, xemachine.reserved_grf_count = 1 : i32} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %wide = xemachine.mov %r0 {execSize = 32 : i32, noMask, src0Region = #xemachine.region<1, 1, 0>} : (!xemachine.reg<16, 0>, i32) -> !xemachine.reg<32, -1>
    return
  }
}

// CHECK: source region exceeds declared register storage

// -----

module {
  func.func @three_grf_source() attributes {
      xemachine.grf_count = 8 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %zero = xemachine.imm 0 : i32
    %storage = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<48, -1>
    %result = xemachine.mov %storage {
        execSize = 32 : i32,
        src0Sub = 8 : i32,
        src0Region = #xemachine.region<1, 1, 0>}
        : (!xemachine.reg<48, -1>, i32) -> !xemachine.reg<32, -1>
    return
  }
}

// CHECK: source 0 region spans more than two GRFs

// -----

module {
  func.func @three_grf_destination() attributes {
      xemachine.grf_count = 8 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %zero = xemachine.imm 0 : i32
    %base = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<48, -1>
    %replacement = xemachine.mov %zero {execSize = 32 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %updated = xemachine.update_tuple %base, %replacement {offsets = [8]}
        : (!xemachine.reg<48, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<48, -1>
    return
  }
}

// CHECK: destination region spans more than two GRFs

// -----

module {
  func.func @source_row_crosses_grf() attributes {
      xemachine.grf_count = 4 : i32,
      xemachine.reserved_grf_count = 0 : i32} {
    %zero = xemachine.imm 0 : i32
    %storage = xemachine.mov %zero {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
    %result = xemachine.mov %storage {
        execSize = 16 : i32,
        src0Sub = 8 : i32,
        src0Region = #xemachine.region<16, 16, 1>}
        : (!xemachine.reg<32, -1>, i32) -> !xemachine.reg<16, -1>
    return
  }
}

// CHECK: source 0 row crosses a GRF boundary
