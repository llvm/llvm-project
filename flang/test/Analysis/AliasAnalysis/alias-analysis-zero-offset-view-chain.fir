// Test that values connected through zero-offset FortranObjectViewOpInterface
// operations (embox, declare, convert) are recognized as MustAlias, even when
// getSource() would set approximateSource due to upstream sliced operations.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Test: embox(ref) vs ref => MustAlias
// CHECK-LABEL: Testing : "_QPtest_embox_ref"
// CHECK-DAG: ref#0 <-> box#0: MustAlias
func.func @_QPtest_embox_ref(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "b"}) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %ref = fir.declare %arg0(%shape) {test.ptr = "ref", uniq_name = "b"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %box = fir.embox %ref(%shape) {test.ptr = "box"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  return
}

// -----

// Test: declare(ref) vs ref => MustAlias
// CHECK-LABEL: Testing : "_QPtest_declare_ref"
// CHECK-DAG: ref#0 <-> decl#0: MustAlias
func.func @_QPtest_declare_ref() {
  %ref = fir.alloca !fir.array<10xf32> {test.ptr = "ref"}
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %decl = fir.declare %ref(%shape) {test.ptr = "decl", uniq_name = "arr"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  return
}

// -----

// Test: embox(declare(ref)) vs ref => MustAlias (two-hop chain)
// CHECK-LABEL: Testing : "_QPtest_embox_declare_ref"
// CHECK-DAG: ref#0 <-> box#0: MustAlias
func.func @_QPtest_embox_declare_ref() {
  %ref = fir.alloca !fir.array<10xf32> {test.ptr = "ref"}
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %decl = fir.declare %ref(%shape) {uniq_name = "arr"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %box = fir.embox %decl(%shape) {test.ptr = "box"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  return
}

// -----

// Test: two box_addr from the same source => MustAlias (sibling case)
// CHECK-LABEL: Testing : "_QPtest_sibling_box_addr"
// CHECK-DAG: addr1#0 <-> addr2#0: MustAlias
func.func @_QPtest_sibling_box_addr(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QPtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %addr1 = fir.box_addr %1#0 {test.ptr = "addr1"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  %addr2 = fir.box_addr %1#0 {test.ptr = "addr2"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  return
}

// -----

// Test: embox with slice does NOT chain-walk to the ref (non-zero offset)
// CHECK-LABEL: Testing : "_QPtest_embox_slice_ref"
// CHECK-DAG: ref#0 <-> sliced_box#0: MayAlias
func.func @_QPtest_embox_slice_ref() {
  %ref = fir.alloca !fir.array<10xf32> {test.ptr = "ref"}
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c5, %c1 : (index, index, index) -> !fir.slice<1>
  %sliced_box = fir.embox %ref(%shape) [%slice] {test.ptr = "sliced_box"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}
