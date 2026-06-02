// Check the ScopedOrigin-based scope-aware refinement in FIR
// AliasAnalysis. After FIR inlining, the AliasAnalysis::getSource
// walk no longer terminates at the original [hl]fir.declare of a
// dummy argument: an inlined call appears as a sequence of fir.box
// / fir.declare / fir.dummy_scope / fir.embox operations that walk
// past the variable's declaration. The scope-aware refinement uses
// per-declare snapshots (ScopedOrigin) collected during the walk so
// that pairs of declares dominated by the same fir.dummy_scope can
// still be disambiguated.
//
// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// -----

// Two distinct, plain (non-TARGET, non-POINTER) dummy arguments of
// the SAME procedure frame: Fortran 2018 15.5.2.13 says they cannot
// alias. Without scope-aware refinement, the walks would terminate
// at the Argument SourceKind but both Sources would have unrelated
// roots and the alias() conservative path would report MayAlias.
//
// CHECK-LABEL: Testing : "_QPtest_two_dummies"
// CHECK-DAG: x#0 <-> y#0: NoAlias
func.func @_QPtest_two_dummies(
    %arg0: !fir.ref<f32> {fir.bindc_name = "x"},
    %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %scope = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %arg0 dummy_scope %scope
      {test.ptr = "x", uniq_name = "_QFtest_two_dummiesEx"}
      : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %y:2 = hlfir.declare %arg1 dummy_scope %scope
      {test.ptr = "y", uniq_name = "_QFtest_two_dummiesEy"}
      : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Distinct dummy and a local of the SAME procedure frame. The local
// is also covered by the function's fir.dummy_scope (it is dominated
// by the unique dummy_scope) and the local is not TARGET, so the
// refinement returns NoAlias.
//
// CHECK-LABEL: Testing : "_QPtest_dummy_vs_local"
// CHECK-DAG: x#0 <-> l#0: NoAlias
func.func @_QPtest_dummy_vs_local(
    %arg0: !fir.ref<f32> {fir.bindc_name = "x"}) {
  %scope = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %arg0 dummy_scope %scope
      {test.ptr = "x", uniq_name = "_QFtest_dummy_vs_localEx"}
      : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %alloca = fir.alloca f32 {bindc_name = "l", uniq_name = "_QFtest_dummy_vs_localEl"}
  %l:2 = hlfir.declare %alloca
      {test.ptr = "l", uniq_name = "_QFtest_dummy_vs_localEl"}
      : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// One of the dummies has TARGET attribute. A pointer or another
// TARGET could reach it, so the refinement must NOT downgrade the
// result to NoAlias; it stays MayAlias.
//
// CHECK-LABEL: Testing : "_QPtest_target_dummy"
// CHECK-DAG: xt#0 <-> y#0: MayAlias
func.func @_QPtest_target_dummy(
    %arg0: !fir.ref<f32> {fir.bindc_name = "xt"},
    %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %scope = fir.dummy_scope : !fir.dscope
  %xt:2 = hlfir.declare %arg0 dummy_scope %scope
      {fortran_attrs = #fir.var_attrs<target>, test.ptr = "xt",
       uniq_name = "_QFtest_target_dummyExt"}
      : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %y:2 = hlfir.declare %arg1 dummy_scope %scope
      {fortran_attrs = #fir.var_attrs<target>, test.ptr = "y",
       uniq_name = "_QFtest_target_dummyEy"}
      : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Same declare on both sides: the scope-aware refinement skips the
// pair when the two snapshots share the same declValue, so the
// original (correct) MustAlias / MayAlias result from the underlying
// alias() is preserved.
//
// CHECK-LABEL: Testing : "_QPtest_same_declare"
// CHECK-DAG: x#0 <-> x#1: MustAlias
func.func @_QPtest_same_declare(
    %arg0: !fir.ref<f32> {fir.bindc_name = "x"}) {
  %scope = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %arg0 dummy_scope %scope
      {test.ptr = "x", uniq_name = "_QFtest_same_declareEx"}
      : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Same as test_two_dummies, but expressed with fir.declare (single
// result) instead of hlfir.declare. This mirrors production FIR after
// convert-hlfir-to-fir, where the scope-aware refinement must still
// disambiguate two plain dummies sharing one fir.dummy_scope.
//
// CHECK-LABEL: Testing : "_QPtest_two_dummies_fir"
// CHECK-DAG: x.fir#0 <-> y.fir#0: NoAlias
func.func @_QPtest_two_dummies_fir(
    %arg0: !fir.ref<f32> {fir.bindc_name = "x"},
    %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %scope = fir.dummy_scope : !fir.dscope
  %x = fir.declare %arg0 dummy_scope %scope
      {test.ptr = "x.fir", uniq_name = "_QFtest_two_dummies_firEx"}
      : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %y = fir.declare %arg1 dummy_scope %scope
      {test.ptr = "y.fir", uniq_name = "_QFtest_two_dummies_firEy"}
      : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  return
}

// -----

// Two-level inlining: the queried values are distinct dummies of the
// INNER (inlined) procedure frame, but each is only reachable through
// a fir.if region-branch introduced in the caller frame (a contiguity
// copy-in for `field`, an OPTIONAL presence select for `value`). The
// else branches yield a different SourceKind (a local temp / fir.absent),
// so the top-level getSource walk for each value merges to
// SourceKind::Unknown and the underlying alias() reports MayAlias
// ("indirect access"). The ONLY way to disambiguate is via the
// ScopedOrigin snapshots taken at the inner-frame declares (which share
// one fir.dummy_scope). This is a regression for buildSourceAtDeclare:
// it must classify each rebuilt Source AT the captured declare's own
// scope (getLastInstantiationPoint=true, yielding SourceKind::Argument)
// rather than walking past the declare back into the fir.if and
// collapsing to Unknown.
//
// CHECK-LABEL: Testing : "_QPtest_nested_inline_region_branch"
// CHECK-DAG: field#0 <-> value#0: NoAlias
func.func @_QPtest_nested_inline_region_branch(
    %arg0: !fir.ref<f32> {fir.bindc_name = "field"},
    %arg1: !fir.ref<f32> {fir.bindc_name = "value", fir.optional},
    %cond: i1) {
  // field: contiguity copy-in select -> original actual or a local temp.
  %ftmp = fir.alloca f32
  %f_sel = fir.if %cond -> (!fir.ref<f32>) {
    fir.result %arg0 : !fir.ref<f32>
  } else {
    fir.result %ftmp : !fir.ref<f32>
  }
  // value: OPTIONAL presence select -> actual or absent.
  %present = fir.is_present %arg1 : (!fir.ref<f32>) -> i1
  %v_sel = fir.if %present -> (!fir.ref<f32>) {
    fir.result %arg1 : !fir.ref<f32>
  } else {
    %absent = fir.absent !fir.ref<f32>
    fir.result %absent : !fir.ref<f32>
  }
  // Inner (inlined) frame: distinct dummies sharing one fir.dummy_scope.
  %inner = fir.dummy_scope : !fir.dscope
  %f_inner = fir.declare %f_sel dummy_scope %inner
      {fortran_attrs = #fir.var_attrs<intent_inout>, test.ptr = "field",
       uniq_name = "_QFinnerEfield"}
      : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %v_inner = fir.declare %v_sel dummy_scope %inner
      {fortran_attrs = #fir.var_attrs<intent_in>, test.ptr = "value",
       uniq_name = "_QFinnerEvalue"}
      : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  return
}
