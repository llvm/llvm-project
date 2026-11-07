// Test that the address of a descriptor and the address of data that does not
// contain a descriptor are disambiguated when the addresses are converted to
// types other than FIR references (e.g. memref), by looking at the FIR types
// they were converted from. The addresses come from an unknown source, so
// that only their types can tell them apart.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading --allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=ALIAS
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' -split-input-file --mlir-disable-threading --allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=MODREF

// ALIAS-LABEL: Testing : "descriptor_vs_data"
// ALIAS: descriptor#0 <-> data#0: NoAlias
// MODREF-LABEL: Testing : "descriptor_vs_data"
// MODREF: store -> data#0: NoModRef
func.func @descriptor_vs_data(%v: i64) {
  %box_ref = "test.source"() : () -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %data_ref = "test.source"() : () -> !fir.ref<f32>
  %descriptor = fir.convert %box_ref {test.ptr = "descriptor"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> memref<i64>
  %data = fir.convert %data_ref {test.ptr = "data"} : (!fir.ref<f32>) -> memref<f32>
  memref.store %v, %descriptor[] {test.ptr = "store"} : memref<i64>
  return
}

// -----

// The data is a derived type with a descriptor component, so it may contain
// the descriptor.
// ALIAS-LABEL: Testing : "descriptor_vs_record_with_descriptor"
// ALIAS: descriptor#0 <-> data#0: MayAlias
// MODREF-LABEL: Testing : "descriptor_vs_record_with_descriptor"
// MODREF: store -> data#0: Mod
func.func @descriptor_vs_record_with_descriptor(%v: i64) {
  %box_ref = "test.source"() : () -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %data_ref = "test.source"() : () -> !fir.ref<!fir.type<t{p:!fir.box<!fir.heap<f32>>}>>
  %descriptor = fir.convert %box_ref {test.ptr = "descriptor"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> memref<i64>
  %data = fir.convert %data_ref {test.ptr = "data"} : (!fir.ref<!fir.type<t{p:!fir.box<!fir.heap<f32>>}>>) -> memref<i64>
  memref.store %v, %descriptor[] {test.ptr = "store"} : memref<i64>
  return
}

// -----

// A view of the converted address still points to the descriptor.
// ALIAS-LABEL: Testing : "descriptor_view_vs_data"
// ALIAS: descriptor#0 <-> data#0: NoAlias
// MODREF-LABEL: Testing : "descriptor_view_vs_data"
// MODREF: store -> data#0: NoModRef
func.func @descriptor_view_vs_data(%v: i64) {
  %box_ref = "test.source"() : () -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %data_ref = "test.source"() : () -> !fir.ref<f32>
  %converted = fir.convert %box_ref : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> memref<i64>
  %descriptor = memref.reinterpret_cast %converted to offset: [0], sizes: [], strides: [] {test.ptr = "descriptor"} : memref<i64> to memref<i64>
  %data = fir.convert %data_ref {test.ptr = "data"} : (!fir.ref<f32>) -> memref<f32>
  memref.store %v, %descriptor[] {test.ptr = "store"} : memref<i64>
  return
}

// -----

// The reduction storage is the converted address of a descriptor.
// ALIAS-LABEL: Testing : "descriptor_reduction_vs_data"
// ALIAS: descriptor#0 <-> data#0: NoAlias
// MODREF-LABEL: Testing : "descriptor_reduction_vs_data"
// MODREF: store -> data#0: NoModRef
func.func @descriptor_reduction_vs_data(%v: i64) {
  %var = "test.source"() : () -> memref<i64>
  %data_ref = "test.source"() : () -> !fir.ref<f32>
  %descriptor = acc.reduction_init %var <add> : memref<i64> {
    %box_ref = "test.source"() : () -> !fir.ref<!fir.box<!fir.heap<f32>>>
    %converted = fir.convert %box_ref : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> memref<i64>
    acc.yield %converted : memref<i64>
  } {test.ptr = "descriptor"}
  %data = fir.convert %data_ref {test.ptr = "data"} : (!fir.ref<f32>) -> memref<f32>
  memref.store %v, %descriptor[] {test.ptr = "store"} : memref<i64>
  return
}
