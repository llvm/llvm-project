// REQUIRES: asserts
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -debug-only=fir-alias-analysis --mlir-disable-threading 2>&1 | FileCheck %s

// Verify that a CUF allocation is recognized as SourceKind::Allocate by
// fir::AliasAnalysis::getSource.

module {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    // Allocate two independent device arrays and tag the results; with
    // value-scoped MemAlloc handling in AA, these should be classified as
    // Allocate and not alias.
    %a = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a1", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa1", test.ptr = "cuf_alloc_a"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %b = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a2", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa2", test.ptr = "cuf_alloc_b"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    return
  }
}

// CHECK-LABEL: Testing : "_QQmain"
// Distinct allocations should not alias.
// CHECK: cuf_alloc_a#0 <-> cuf_alloc_b#0: NoAlias


