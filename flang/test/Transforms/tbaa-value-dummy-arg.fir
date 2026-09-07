// Verify that AddAliasTags tags a VALUE dummy argument copy-store under
// "dummy arg data" (not "allocated data"), preventing DSE from eliminating it.
//
// Flang lowers a large VALUE dummy arg passed by value as:
//
//   %arg0: !fir.type<T>                    <- struct value
//   %alloca = fir.alloca T
//   fir.store %arg0 to %alloca             <- copy-store  [pre-fix: "allocated data"]
//   %decl = fir.declare %alloca dummy_scope %scope arg 1
//              {fortran_attrs = #fir.var_attrs<value>}
//   loop: fir.load via fir.array_coor %decl  <- reads  ["dummy arg data/_QFfnt4Edt"]
//
// Pre-fix: the copy-store gets SourceKind::Allocate -> "allocated data"; the
// loop reads get SourceKind::Argument -> "dummy arg data/_QFfnt4Edt".  These
// are sibling subtrees -> TBAA NoAlias -> DSE removes the copy-store even
// though the loop reads (returned as the function result) are live.
//
// The fix tags the copy-store under "dummy arg data" (ancestor of the leaf),
// so both accesses are in the same subtree and DSE leaves the store alone.
//
// See flang/test/Fir/tbaa-value-dummy-arg-dse.ll for the DSE consequence.
//
// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

// "dummy arg data" intermediate node must exist.
// CHECK-DAG: #[[DUMMY_NODE:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members

// A tag rooted at DUMMY_NODE must exist (the post-fix copy-store tag).
// Pre-fix: no such tag (copy-store uses "allocated data" instead).
// CHECK-DAG: #[[COPY_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMY_NODE]], access_type = #[[DUMMY_NODE]],

// The per-variable leaf must descend from DUMMY_NODE.
// CHECK-DAG: #[[DUMMY_LEAF:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFfnt4Edt", members = {<#[[DUMMY_NODE]], 0>}>

// The loop-read tag must use the leaf.
// CHECK-DAG: #[[LEAF_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMY_LEAF]], access_type = #[[DUMMY_LEAF]],

// CHECK-LABEL: func.func @fnt4(
// The copy-store must use the dummy-arg base tag, not "allocated data".
// CHECK: fir.store %arg0 {{.*}} {tbaa = [#[[COPY_TAG]]]}
// The loop read must use the dummy-arg leaf tag.
// CHECK: fir.load {{.*}} {tbaa = [#[[LEAF_TAG]]]}

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    !llvm.ptr = dense<64> : vector<4xi64>,
    i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>,
    i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>,
    i64 = dense<[32, 64]> : vector<2xi64>,
    f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>,
    f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">,
  llvm.data_layout = ""} {

// Reduced from PR #215366.  integer(1) function fnt4(dt) bind(c) with
// type(dtd1), value :: dt and a loop summing dt%a(i) -- result is returned
// so loop reads are live; DSE misfire is purely TBAA-driven, not liveness.

func.func @fnt4(%arg0: !fir.type<_QMmTdtd1{a:!fir.array<50xi8>}> {fir.bindc_name = "dt"}) -> i8
    attributes {fir.bindc_name = "fnt4", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
  %c0_i8  = arith.constant 0  : i8
  %c1     = arith.constant 1  : index
  %c10    = arith.constant 10 : index
  %c1_i64 = arith.constant 1  : i64
  %scope  = fir.dummy_scope : !fir.dscope
  %alloca = fir.alloca !fir.type<_QMmTdtd1{a:!fir.array<50xi8>}>
  // Copy-store: pre-fix "allocated data", post-fix "dummy arg data".
  fir.store %arg0 to %alloca : !fir.ref<!fir.type<_QMmTdtd1{a:!fir.array<50xi8>}>>
  // dummy_scope + arg 1 + value causes loop reads to get SourceKind::Argument.
  %decl = fir.declare %alloca dummy_scope %scope arg 1
            {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFfnt4Edt"} :
          (!fir.ref<!fir.type<_QMmTdtd1{a:!fir.array<50xi8>}>>, !fir.dscope) ->
           !fir.ref<!fir.type<_QMmTdtd1{a:!fir.array<50xi8>}>>
  // Accumulator -- genuine SourceKind::Allocate, unaffected by the fix.
  %result_alloca = fir.alloca i8 {uniq_name = "_QFfnt4Efnt4"}
  %result_decl   = fir.declare %result_alloca {uniq_name = "_QFfnt4Efnt4"} :
                   (!fir.ref<i8>) -> !fir.ref<i8>
  fir.store %c0_i8 to %result_decl : !fir.ref<i8>
  // Loop reads: "dummy arg data/_QFfnt4Edt".  Live -- result is returned.
  %shape   = fir.shape %c10 : (index) -> !fir.shape<1>
  %field_a = fir.coordinate_of %decl, a :
      (!fir.ref<!fir.type<_QMmTdtd1{a:!fir.array<50xi8>}>>) -> !fir.ref<!fir.array<50xi8>>
  %loop_result = fir.do_loop %iv = %c1 to %c10 step %c1
                   iter_args(%acc = %c0_i8) -> i8 {
    %iv_i64   = fir.convert %iv : (index) -> i64
    %elem_ref = fir.array_coor %field_a(%shape) %iv_i64 :
        (!fir.ref<!fir.array<50xi8>>, !fir.shape<1>, i64) -> !fir.ref<i8>
    %elem = fir.load %elem_ref : !fir.ref<i8>
    %next = arith.addi %acc, %elem : i8
    fir.result %next : i8
  }
  return %loop_result : i8
}
}
