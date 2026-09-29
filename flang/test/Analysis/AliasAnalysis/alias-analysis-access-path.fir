// Edge-case tests for the AccessPath infrastructure in fir::AliasAnalysis.
//
// These tests exercise scenarios that may arise when the IR is not in the
// canonical form produced by lowering (e.g. missing fir.declare, non-FIR ops
// in the def-use chain) and verify that alias analysis remains sound.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading  \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// -----

// Bare fir.alloca without fir.declare.
// Component disambiguation must still work because fir.coordinate_of sets
// approximateSource (its getViewOffset returns nullopt), which triggers
// pathsDivergeAtComponent.

// CHECK-LABEL: Testing : "_QPtest_bare_alloca"
// CHECK-DAG: comp_a#0 <-> comp_b#0: NoAlias
// CHECK-DAG: comp_a#0 <-> comp_a2#0: MayAlias
// CHECK-DAG: comp_b#0 <-> comp_a2#0: NoAlias

func.func @_QPtest_bare_alloca() {
  %0 = fir.alloca !fir.type<_QMmTty{a:f32,b:f32}> {uniq_name = "_QFtestEobj"}
  %a = fir.coordinate_of %0, a {test.ptr = "comp_a"} : (!fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>) -> !fir.ref<f32>
  %b = fir.coordinate_of %0, b {test.ptr = "comp_b"} : (!fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>) -> !fir.ref<f32>
  %a2 = fir.coordinate_of %0, a {test.ptr = "comp_a2"} : (!fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %a : !fir.ref<f32>
  %v2 = fir.load %b : !fir.ref<f32>
  %v3 = fir.load %a2 : !fir.ref<f32>
  return
}

// -----

// Non-FIR operation (arith.select) breaking the def-use chain.
// The walk hits the Default handler, producing SourceKind::Unknown, so all
// pairs involving the broken chain must conservatively return MayAlias.
// The direct access through the original alloca is also MayAlias with the
// broken-chain accesses because their origins differ (Unknown vs Allocate).

// CHECK-LABEL: Testing : "_QPtest_unknown_op"
// CHECK-DAG: sel_comp_a#0 <-> sel_comp_b#0: MayAlias
// CHECK-DAG: sel_comp_a#0 <-> direct_comp_a#0: MayAlias
// CHECK-DAG: sel_comp_b#0 <-> direct_comp_a#0: MayAlias

func.func @_QPtest_unknown_op(%cond: i1) {
  %0 = fir.alloca !fir.type<_QMmTty{a:f32,b:f32}> {uniq_name = "_QFtestEobj1"}
  %1 = fir.alloca !fir.type<_QMmTty{a:f32,b:f32}> {uniq_name = "_QFtestEobj2"}
  %sel = arith.select %cond, %0, %1 : !fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>
  %a = fir.coordinate_of %sel, a {test.ptr = "sel_comp_a"} : (!fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>) -> !fir.ref<f32>
  %b = fir.coordinate_of %sel, b {test.ptr = "sel_comp_b"} : (!fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>) -> !fir.ref<f32>
  %direct_a = fir.coordinate_of %0, a {test.ptr = "direct_comp_a"} : (!fir.ref<!fir.type<_QMmTty{a:f32,b:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %a : !fir.ref<f32>
  %v2 = fir.load %b : !fir.ref<f32>
  %v3 = fir.load %direct_a : !fir.ref<f32>
  return
}

// -----

// Dynamic array index in fir.coordinate_of must NOT produce a Component step.
// Two subscripts of the same array component are MayAlias (could be the same
// element), but an array element vs a different named component is NoAlias.

// CHECK-LABEL: Testing : "_QPtest_dynamic_index"
// CHECK-DAG: arr_elem1#0 <-> arr_elem2#0: MayAlias
// CHECK-DAG: arr_elem1#0 <-> comp_b#0: NoAlias
// CHECK-DAG: arr_elem2#0 <-> comp_b#0: NoAlias

func.func @_QPtest_dynamic_index(%idx1: index, %idx2: index) {
  %0 = fir.alloca !fir.type<_QMmTty2{arr:!fir.array<10xf32>,b:f32}> {uniq_name = "_QFtestEobj"}
  %obj = fir.declare %0 {uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTty2{arr:!fir.array<10xf32>,b:f32}>>) -> !fir.ref<!fir.type<_QMmTty2{arr:!fir.array<10xf32>,b:f32}>>
  %arrref = fir.coordinate_of %obj, arr : (!fir.ref<!fir.type<_QMmTty2{arr:!fir.array<10xf32>,b:f32}>>) -> !fir.ref<!fir.array<10xf32>>
  %elem1 = fir.coordinate_of %arrref, %idx1 {test.ptr = "arr_elem1"} : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  %elem2 = fir.coordinate_of %arrref, %idx2 {test.ptr = "arr_elem2"} : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  %bref = fir.coordinate_of %obj, b {test.ptr = "comp_b"} : (!fir.ref<!fir.type<_QMmTty2{arr:!fir.array<10xf32>,b:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %elem1 : !fir.ref<f32>
  %v2 = fir.load %elem2 : !fir.ref<f32>
  %v3 = fir.load %bref : !fir.ref<f32>
  return
}

// -----

// Nested derived type: multi-level component paths.
// x%inner%a and x%inner%b diverge at the second Component step -> NoAlias.
// x%inner%a and x%j diverge at the first Component step -> NoAlias.

// CHECK-LABEL: Testing : "_QPtest_nested"
// CHECK-DAG: inner_a#0 <-> inner_b#0: NoAlias
// CHECK-DAG: inner_a#0 <-> outer_j#0: NoAlias
// CHECK-DAG: inner_b#0 <-> outer_j#0: NoAlias

func.func @_QPtest_nested() {
  %0 = fir.alloca !fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}> {uniq_name = "_QFtestEobj"}
  %obj = fir.declare %0 {uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>
  %inner = fir.coordinate_of %obj, inner : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<!fir.type<_QMmTinner{a:f32,b:f32}>>
  %a = fir.coordinate_of %inner, a {test.ptr = "inner_a"} : (!fir.ref<!fir.type<_QMmTinner{a:f32,b:f32}>>) -> !fir.ref<f32>
  %b = fir.coordinate_of %inner, b {test.ptr = "inner_b"} : (!fir.ref<!fir.type<_QMmTinner{a:f32,b:f32}>>) -> !fir.ref<f32>
  %j = fir.coordinate_of %obj, j {test.ptr = "outer_j"} : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %a : !fir.ref<f32>
  %v2 = fir.load %b : !fir.ref<f32>
  %v3 = fir.load %j : !fir.ref<f32>
  return
}

// -----

// Multi-field fir.coordinate_of accessing nested fields in a single operation.
// The access path must record a PathStep for each field index.

// CHECK-LABEL: Testing : "_QPtest_multi_field_coord"
// CHECK-DAG: mf_inner_a#0 <-> mf_inner_b#0: NoAlias
// CHECK-DAG: mf_inner_a#0 <-> mf_outer_j#0: NoAlias
// CHECK-DAG: mf_inner_b#0 <-> mf_outer_j#0: NoAlias

func.func @_QPtest_multi_field_coord() {
  %0 = fir.alloca !fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}> {uniq_name = "_QFtestEobj"}
  %obj = fir.declare %0 {uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>
  %a = fir.coordinate_of %obj, inner, a {test.ptr = "mf_inner_a"} : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<f32>
  %b = fir.coordinate_of %obj, inner, b {test.ptr = "mf_inner_b"} : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<f32>
  %j = fir.coordinate_of %obj, j {test.ptr = "mf_outer_j"} : (!fir.ref<!fir.type<_QMmTouter{inner:!fir.type<_QMmTinner{a:f32,b:f32}>,j:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %a : !fir.ref<f32>
  %v2 = fir.load %b : !fir.ref<f32>
  %v3 = fir.load %j : !fir.ref<f32>
  return
}

// -----

// Interleaved component and 1-D array index in a single fir.coordinate_of.
// The dynamic index (kDynamicIndex) must be consumed by the dimension peeling
// logic without producing a Component step.
// obj%arr_of_inner(idx)%a and obj%arr_of_inner(idx)%b diverge at the leaf
// component -> NoAlias.  Both diverge from obj%j at the first step -> NoAlias.

// CHECK-LABEL: Testing : "_QPtest_interleaved_1d"
// CHECK-DAG: arr_inner_a#0 <-> arr_inner_b#0: NoAlias
// CHECK-DAG: arr_inner_a#0 <-> outer_j#0: NoAlias
// CHECK-DAG: arr_inner_b#0 <-> outer_j#0: NoAlias

func.func @_QPtest_interleaved_1d(%idx1: index, %idx2: index) {
  %0 = fir.alloca !fir.type<_QMmTwrap{arr_of_inner:!fir.array<10x!fir.type<_QMmTinner{a:f32,b:f32}>>,j:f32}> {uniq_name = "_QFtestEobj"}
  %obj = fir.declare %0 {uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTwrap{arr_of_inner:!fir.array<10x!fir.type<_QMmTinner{a:f32,b:f32}>>,j:f32}>>) -> !fir.ref<!fir.type<_QMmTwrap{arr_of_inner:!fir.array<10x!fir.type<_QMmTinner{a:f32,b:f32}>>,j:f32}>>
  %a = fir.coordinate_of %obj, arr_of_inner, %idx1, a {test.ptr = "arr_inner_a"} : (!fir.ref<!fir.type<_QMmTwrap{arr_of_inner:!fir.array<10x!fir.type<_QMmTinner{a:f32,b:f32}>>,j:f32}>>, index) -> !fir.ref<f32>
  %b = fir.coordinate_of %obj, arr_of_inner, %idx2, b {test.ptr = "arr_inner_b"} : (!fir.ref<!fir.type<_QMmTwrap{arr_of_inner:!fir.array<10x!fir.type<_QMmTinner{a:f32,b:f32}>>,j:f32}>>, index) -> !fir.ref<f32>
  %j = fir.coordinate_of %obj, j {test.ptr = "outer_j"} : (!fir.ref<!fir.type<_QMmTwrap{arr_of_inner:!fir.array<10x!fir.type<_QMmTinner{a:f32,b:f32}>>,j:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %a : !fir.ref<f32>
  %v2 = fir.load %b : !fir.ref<f32>
  %v3 = fir.load %j : !fir.ref<f32>
  return
}

// -----

// Nested 2-D arrays: component, idx, idx, component, idx, idx.
// The dimension peeling loop must fire twice (once per array level), each
// time consuming two kDynamicIndex entries before advancing to the element type.
// obj%arr_of_inner(i,j)%x(k,l) and obj%arr_of_inner(i,j)%y(k,l) diverge at
// the second component -> NoAlias.

// CHECK-LABEL: Testing : "_QPtest_nested_2d_arrays"
// CHECK-DAG: nested_x#0 <-> nested_y#0: NoAlias
// CHECK-DAG: nested_x#0 <-> outer_k#0: NoAlias
// CHECK-DAG: nested_y#0 <-> outer_k#0: NoAlias

func.func @_QPtest_nested_2d_arrays(%idx1: index, %idx2: index, %idx3: index, %idx4: index, %idx5: index, %idx6: index, %idx7: index, %idx8: index) {
  %0 = fir.alloca !fir.type<_QMmTouter2{arr_of_inner:!fir.array<10x20x!fir.type<_QMmTinner2{x:!fir.array<5x10xf32>,y:!fir.array<5x10xf32>}>>,k:f32}> {uniq_name = "_QFtestEobj"}
  %obj = fir.declare %0 {uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTouter2{arr_of_inner:!fir.array<10x20x!fir.type<_QMmTinner2{x:!fir.array<5x10xf32>,y:!fir.array<5x10xf32>}>>,k:f32}>>) -> !fir.ref<!fir.type<_QMmTouter2{arr_of_inner:!fir.array<10x20x!fir.type<_QMmTinner2{x:!fir.array<5x10xf32>,y:!fir.array<5x10xf32>}>>,k:f32}>>
  %x = fir.coordinate_of %obj, arr_of_inner, %idx1, %idx2, x, %idx3, %idx4 {test.ptr = "nested_x"} : (!fir.ref<!fir.type<_QMmTouter2{arr_of_inner:!fir.array<10x20x!fir.type<_QMmTinner2{x:!fir.array<5x10xf32>,y:!fir.array<5x10xf32>}>>,k:f32}>>, index, index, index, index) -> !fir.ref<f32>
  %y = fir.coordinate_of %obj, arr_of_inner, %idx5, %idx6, y, %idx7, %idx8 {test.ptr = "nested_y"} : (!fir.ref<!fir.type<_QMmTouter2{arr_of_inner:!fir.array<10x20x!fir.type<_QMmTinner2{x:!fir.array<5x10xf32>,y:!fir.array<5x10xf32>}>>,k:f32}>>, index, index, index, index) -> !fir.ref<f32>
  %k = fir.coordinate_of %obj, k {test.ptr = "outer_k"} : (!fir.ref<!fir.type<_QMmTouter2{arr_of_inner:!fir.array<10x20x!fir.type<_QMmTinner2{x:!fir.array<5x10xf32>,y:!fir.array<5x10xf32>}>>,k:f32}>>) -> !fir.ref<f32>
  %v1 = fir.load %x : !fir.ref<f32>
  %v2 = fir.load %y : !fir.ref<f32>
  %v3 = fir.load %k : !fir.ref<f32>
  return
}
