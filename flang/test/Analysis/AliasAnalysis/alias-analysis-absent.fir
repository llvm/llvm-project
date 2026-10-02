// Exercise FIR alias analysis for null addresses. fir.absent and an
// address-typed fir.zero_bits are nulls: each is associated with no
// address, so it does not alias real storage, another null, or a global.
// It aliases only itself.

// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "_QPtest_absent"
// CHECK-DAG: absent1#0 <-> absent2#0: NoAlias
// CHECK-DAG: real#0 <-> absent1#0: NoAlias
// CHECK-DAG: real#0 <-> absent2#0: NoAlias

func.func @_QPtest_absent() {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %alloc = fir.alloca !fir.array<10xf32>
  %real = fir.declare %alloc(%shape) {uniq_name = "_QPtest_absentEx", test.ptr = "real"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %abs1 = fir.absent !fir.ref<!fir.array<10xf32>> {test.ptr = "absent1"}
  %abs2 = fir.absent !fir.ref<!fir.array<10xf32>> {test.ptr = "absent2"}
  return
}

// -----

// Same fir.absent via fir.convert and fir.declare
// CHECK-LABEL: Testing : "_QPtest_absent_same_through_convert"
// CHECK-DAG: abs_slot#0 <-> via_convert#0: MustAlias
// CHECK-DAG: abs_slot#0 <-> direct_abs#0: MustAlias
// CHECK-DAG: via_convert#0 <-> direct_abs#0: MustAlias

func.func @_QPtest_absent_same_through_convert() {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %abs = fir.absent !fir.ref<!fir.array<10xf32>> {test.ptr = "abs_slot"}
  %conv = fir.convert %abs : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>
  %via = fir.declare %conv(%shape) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QPtest_absent_sameEv", test.ptr = "via_convert"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %direct = fir.declare %abs(%shape) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QPtest_absent_sameEd", test.ptr = "direct_abs"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  return
}

// -----

// Global array storage must not alias an absent.
// CHECK-LABEL: Testing : "_QPtest_absent_vs_global"
// CHECK-DAG: {{(absent_g#0 <-> global#0|global#0 <-> absent_g#0)}}: NoAlias

fir.global @_QPtest_absent_vs_global_g : !fir.array<10xf32> {
  %0 = fir.undefined !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @_QPtest_absent_vs_global() {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %gaddr = fir.address_of(@_QPtest_absent_vs_global_g) : !fir.ref<!fir.array<10xf32>>
  %g = fir.declare %gaddr(%shape) {uniq_name = "_QPtest_absent_vs_globalEg", test.ptr = "global"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %abs = fir.absent !fir.ref<!fir.array<10xf32>> {test.ptr = "absent_g"}
  return
}

// -----

// An address-typed fir.zero_bits is a null address. It does not alias an
// allocation, and two distinct nulls do not alias each other.
// CHECK-LABEL: Testing : "test_null_zero_bits"
// CHECK-DAG: null_a#0 <-> heap#0: NoAlias
// CHECK-DAG: null_a#0 <-> null_b#0: NoAlias

func.func @test_null_zero_bits() {
  %z1 = fir.zero_bits !fir.ref<i32>
  %n1 = fir.convert %z1 {test.ptr = "null_a"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %z2 = fir.zero_bits !fir.ref<i32>
  %n2 = fir.convert %z2 {test.ptr = "null_b"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %a = fir.alloca i32 {uniq_name = "_QFEnull_heap"}
  %d = fir.declare %a {uniq_name = "_QFEnull_heap", test.ptr = "heap"} : (!fir.ref<i32>) -> !fir.ref<i32>
  return
}

// -----

// Two zero-offset views of one address-typed fir.zero_bits. The test harness
// compares distinct values, so each view is a fir.convert of the same null.
// CHECK-LABEL: Testing : "test_null_same_value"
// CHECK-DAG: null_view_a#0 <-> null_view_b#0: MustAlias

func.func @test_null_same_value() {
  %z = fir.zero_bits !fir.ref<i32>
  %a = fir.convert %z {test.ptr = "null_view_a"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %b = fir.convert %z {test.ptr = "null_view_b"} : (!fir.ref<i32>) -> !fir.ref<i32>
  return
}
