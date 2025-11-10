// REQUIRES: asserts
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -debug-only=fir-alias-analysis --mlir-disable-threading 2>&1 | FileCheck %s

// Verify that CUF allocation via cuf.allocate is recognized as
// SourceKind::Allocate by fir::AliasAnalysis::getSource on the box value.

module {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    // Create two independent device boxes and tag their refs.
    %a = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a1", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa1", test.ptr = "cuf_allocate_a"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %b = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a2", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa2", test.ptr = "cuf_allocate_b"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    // Allocate device data for each descriptor; AA should classify the box
    // values (tagged above) as Allocate sources and not alias.
    %sa = cuf.allocate %a : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
    %sb = cuf.allocate %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
    return
  }
}

// CHECK-LABEL: Testing : "_QQmain"
// Distinct allocations via cuf.allocate should not alias.
// CHECK: cuf_allocate_a#0 <-> cuf_allocate_b#0: NoAlias


