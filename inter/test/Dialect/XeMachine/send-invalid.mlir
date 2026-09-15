// RUN: not inter-opt %s --split-input-file --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' 2>&1 | FileCheck %s

func.func @source_descriptor_mismatch() attributes {
    xemachine.grf_count = 128 : i32,
    xemachine.reserved_grf_count = 0 : i32} {
  %payload = xemachine.archreg 4 : !xemachine.reg<32, 4>
  %dst, %token = xemachine.send ugm %payload {
      desc = 34079235 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
      : (!xemachine.reg<32, 4>)
      -> (!xemachine.reg<0, 0>, !xemachine.mem.token)
  return
}

// -----

func.func @destination_descriptor_mismatch() attributes {
    xemachine.grf_count = 128 : i32,
    xemachine.reserved_grf_count = 0 : i32} {
  %payload = xemachine.archreg 4 : !xemachine.reg<16, 4>
  %dst, %token = xemachine.send ugm %payload {
      desc = 37749251 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
      : (!xemachine.reg<16, 4>)
      -> (!xemachine.reg<16, 5>, !xemachine.mem.token)
  return
}

// -----

func.func @oversized_source_1() attributes {
    xemachine.grf_count = 128 : i32,
    xemachine.reserved_grf_count = 0 : i32} {
  %payload = xemachine.archreg 4 : !xemachine.reg<16, 4>
  %data = xemachine.archreg 5 : !xemachine.reg<512, 5>
  %dst, %token = xemachine.send ugm %payload data %data {
      desc = 34079235 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
      : (!xemachine.reg<16, 4>, !xemachine.reg<512, 5>)
      -> (!xemachine.reg<0, 0>, !xemachine.mem.token)
  return
}

// -----

func.func @oversized_named_sources() attributes {
    xemachine.grf_count = 128 : i32,
    xemachine.reserved_grf_count = 0 : i32} {
  %root = xemachine.token
  %address = xemachine.archreg 4 : !xemachine.reg<64, 4>
  %data = xemachine.archreg 8 : !xemachine.reg<448, 8>
  %token = xemachine.store_a64 %address data %data dep %root
      {execSize = 32 : i32}
      : (!xemachine.reg<64, 4>, !xemachine.reg<448, 8>)
      -> !xemachine.mem.token
  return
}

// CHECK: source 0 width does not match the descriptor
// CHECK: destination width does not match the descriptor
// CHECK: source 1 exceeds the 31-GRF encoding limit
// CHECK: combined source payload exceeds 31 GRFs
