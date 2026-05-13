// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// -----

// Distinct fir.alloca in each branch; unrelated fir.alloca outside the if.
// CHECK-LABEL: Testing : "test_rb_both_alloc_distinct"
// CHECK-DAG: outside_alloc#0 <-> join_alloc#0: NoAlias

func.func @test_rb_both_alloc_distinct() {
  %cond = arith.constant true
  %a_out = fir.alloca f32 {uniq_name = "_QFEa_out"}
  %d_out = fir.declare %a_out {uniq_name = "_QFEa_out", test.ptr = "outside_alloc"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    %a1 = fir.alloca f32 {uniq_name = "_QFEa1"}
    %d1 = fir.declare %a1 {uniq_name = "_QFEa1"} : (!fir.ref<f32>) -> !fir.ref<f32>
    fir.result %d1 : !fir.ref<f32>
  } else {
    %a2 = fir.alloca f32 {uniq_name = "_QFEa2"}
    %d2 = fir.declare %a2 {uniq_name = "_QFEa2"} : (!fir.ref<f32>) -> !fir.ref<f32>
    fir.result %d2 : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_alloc"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

// Each branch yields different fir.declare which means that the
// origin is from either - this case is handled as Indirect.
// CHECK-LABEL: Testing : "test_rb_both_alloc_distinct_outer_scope"
// CHECK-DAG: d1#0 <-> join_outer#0: MayAlias
// CHECK-DAG: d2#0 <-> join_outer#0: MayAlias
// CHECK-DAG: outside_outer#0 <-> join_outer#0: MayAlias

func.func @test_rb_both_alloc_distinct_outer_scope() {
  %cond = arith.constant true
  %a1 = fir.alloca f32 {uniq_name = "_QFEa1o"}
  %a2 = fir.alloca f32 {uniq_name = "_QFEa2o"}
  %d1 = fir.declare %a1 {uniq_name = "_QFEa1o", test.ptr = "d1"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %d2 = fir.declare %a2 {uniq_name = "_QFEa2o", test.ptr = "d2"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %a_ext = fir.alloca f32 {uniq_name = "_QFEa_ext_o"}
  %d_ext = fir.declare %a_ext {uniq_name = "_QFEa_ext_o", test.ptr = "outside_outer"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    fir.result %d1 : !fir.ref<f32>
  } else {
    fir.result %d2 : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_outer"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

// Both branches yield the same fir.declare from outside.
// CHECK-LABEL: Testing : "test_rb_both_alloc_same_decl"
// CHECK-DAG: shared_decl#0 <-> join_same#0: MustAlias

func.func @test_rb_both_alloc_same_decl() {
  %cond = arith.constant true
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa", test.ptr = "shared_decl"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    fir.result %da : !fir.ref<f32>
  } else {
    fir.result %da : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_same"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

// An example of conservative behavior where one branch return value comes from
// outside. Because at the merge we do not find a common origin and since there's
// currently no way to record multiple origins, we fallback to MayAlias.
// TODO: Ideally this should be NoAlias
// CHECK-LABEL: Testing : "test_rb_alloc_outer_vs_inner"
// CHECK-DAG: outside_neither#0 <-> join_oi#0: MayAlias

func.func @test_rb_alloc_outer_vs_inner() {
  %cond = arith.constant true
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %a_neither = fir.alloca f32 {uniq_name = "_QFEa_neither"}
  %d_neither = fir.declare %a_neither {uniq_name = "_QFEa_neither", test.ptr = "outside_neither"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    fir.result %da : !fir.ref<f32>
  } else {
    %a_else = fir.alloca f32 {uniq_name = "_QFEa_else"}
    %d_else = fir.declare %a_else {uniq_name = "_QFEa_else"} : (!fir.ref<f32>) -> !fir.ref<f32>
    fir.result %d_else : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_oi"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

fir.global @rb_merge_g : f32 {
  %0 = arith.constant 0.0 : f32
  fir.has_value %0 : f32
}

// Both branches yield the same global.
// CHECK-LABEL: Testing : "test_rb_both_same_global"
// CHECK-DAG: global_decl#0 <-> join_g#0: MustAlias

func.func @test_rb_both_same_global() {
  %cond = arith.constant true
  %addr = fir.address_of(@rb_merge_g) : !fir.ref<f32>
  %dg = fir.declare %addr {uniq_name = "_QErb_merge_g", test.ptr = "global_decl"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    fir.result %dg : !fir.ref<f32>
  } else {
    fir.result %dg : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_g"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

// Both branches yield the same dummy argument.
// CHECK-LABEL: Testing : "test_rb_both_same_argument"
// CHECK-DAG: decl_arg#0 <-> join_arg#0: MustAlias

func.func @test_rb_both_same_argument(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) {
  %cond = arith.constant true
  %ds = fir.dummy_scope : !fir.dscope
  %dx = fir.declare %arg0 dummy_scope %ds arg 1 {uniq_name = "_QFEex", test.ptr = "decl_arg"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    fir.result %dx : !fir.ref<f32>
  } else {
    fir.result %dx : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_arg"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

fir.global @rb_side_g : f32 {
  %0 = arith.constant 0.0 : f32
  fir.has_value %0 : f32
}

// Merged kinds differ (Allocate vs Global) thus conservative MayAlias
// (and this would be the case even if we tracked multiple origins).
// CHECK-LABEL: Testing : "test_rb_merged_unknown_global_vs_alloc"
// CHECK-DAG: outside_mixed#0 <-> join_mixed#0: MayAlias

func.func @test_rb_merged_unknown_global_vs_alloc() {
  %cond = arith.constant true
  %addr = fir.address_of(@rb_side_g) : !fir.ref<f32>
  %dg = fir.declare %addr {uniq_name = "_QErb_side_g"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %a_by = fir.alloca f32 {uniq_name = "_QFEa_by"}
  %d_by = fir.declare %a_by {uniq_name = "_QFEa_by", test.ptr = "outside_mixed"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    fir.result %dg : !fir.ref<f32>
  } else {
    fir.result %da : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_mixed"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

// Distinct fir.alloca in each branch with mismatched Fortran attrs on the declares;
// despite being different allocations inside the branches, we give conservative
// response. This is primarily to test that the attribute merge is working but
// this pattern is unlikely to be generated from real Fortran code.
// CHECK-LABEL: Testing : "test_rb_merged_unknown_attr_mismatch"
// CHECK-DAG: outside_attr#0 <-> join_attr#0: MayAlias

func.func @test_rb_merged_unknown_attr_mismatch() {
  %cond = arith.constant true
  %a_ext = fir.alloca f32 {uniq_name = "_QFEa_attr_ext"}
  %d_ext = fir.declare %a_ext {uniq_name = "_QFEa_attr_ext", test.ptr = "outside_attr"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %jf = fir.if %cond -> !fir.ref<f32> {
    %a1 = fir.alloca f32 {uniq_name = "_QFEa1"}
    %dt = fir.declare %a1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEa1"} : (!fir.ref<f32>) -> !fir.ref<f32>
    fir.result %dt : !fir.ref<f32>
  } else {
    %a2 = fir.alloca f32 {uniq_name = "_QFEa2"}
    %dp = fir.declare %a2 {uniq_name = "_QFEa2"} : (!fir.ref<f32>) -> !fir.ref<f32>
    fir.result %dp : !fir.ref<f32>
  }
  %join = fir.convert %jf {test.ptr = "join_attr"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// -----

// One branch yields the dummy box, the other yields fir.pack_array of the same
// box (approximateSource on that predecessor). Join is disambiguated from a
// different dummy argument's data.
// CHECK-LABEL: Testing : "test_rb_merge_one_branch_pack_array"
// CHECK-DAG: join_pack#0 <-> other_arg_box#0: NoAlias

func.func @test_rb_merge_one_branch_pack_array(
    %arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
    %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
  %c = arith.constant true
  %packed = fir.pack_array %arg0 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %jf = fir.if %c -> !fir.box<!fir.array<?xf32>> {
    fir.result %arg0 : !fir.box<!fir.array<?xf32>>
  } else {
    fir.result %packed : !fir.box<!fir.array<?xf32>>
  }
  %join = fir.convert %jf {test.ptr = "join_pack"} : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %other = fir.convert %arg1 {test.ptr = "other_arg_box"} : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

// OPTIONAL pattern tested against an unrelated fir.alloca.
// CHECK-LABEL: Testing : "test_rb_optional_ref_present_absent"
// CHECK-DAG: outside_opt#0 <-> opt_join#0: NoAlias

func.func @test_rb_optional_ref_present_absent(%arg0: !fir.ref<i32> {fir.bindc_name = "x", fir.optional}) {
  %present = fir.is_present %arg0 : (!fir.ref<i32>) -> i1
  %a_ext = fir.alloca i32 {uniq_name = "_QFEopt_ext"}
  %d_ext = fir.declare %a_ext {uniq_name = "_QFEopt_ext", test.ptr = "outside_opt"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %slot = fir.if %present -> !fir.ref<i32> {
    %a = fir.alloca i32 {uniq_name = "_QFEopt"}
    %d = fir.declare %a {uniq_name = "_QFEopt"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.result %d : !fir.ref<i32>
  } else {
    %abs = fir.absent !fir.ref<i32>
    fir.result %abs : !fir.ref<i32>
  }
  %join = fir.convert %slot {test.ptr = "opt_join"} : (!fir.ref<i32>) -> !fir.ref<i32>
  return
}

// -----

// Same OPTIONAL idea as above, but the unrelated fir.alloca outside the if is
// declared with TARGET: join should still not alias that unrelated storage.
// CHECK-LABEL: Testing : "test_rb_optional_ref_outside_target"
// CHECK-DAG: outside_opt_tgt#0 <-> opt_join_tgt#0: NoAlias

func.func @test_rb_optional_ref_outside_target(%arg0: !fir.ref<i32> {fir.bindc_name = "x", fir.optional}) {
  %present = fir.is_present %arg0 : (!fir.ref<i32>) -> i1
  %a_ext = fir.alloca i32 {uniq_name = "_QFEopt_ext_t"}
  %d_ext = fir.declare %a_ext {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEopt_ext_t", test.ptr = "outside_opt_tgt"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %slot = fir.if %present -> !fir.ref<i32> {
    %a = fir.alloca i32 {uniq_name = "_QFEopt_t"}
    %d = fir.declare %a {uniq_name = "_QFEopt_t"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.result %d : !fir.ref<i32>
  } else {
    %abs = fir.absent !fir.ref<i32>
    fir.result %abs : !fir.ref<i32>
  }
  %join = fir.convert %slot {test.ptr = "opt_join_tgt"} : (!fir.ref<i32>) -> !fir.ref<i32>
  return
}

// -----

// Add loop test whose result is based on BlockArguments to ensure
// no cycle.
// CHECK-LABEL: Testing : "test_rb_do_loop_iter_carry_ref"
// CHECK-DAG: carry_init#0 <-> outside_loop#0: NoAlias
// CHECK-DAG: carry_init#0 <-> loop_join_ref#0: MayAlias
// CHECK-DAG: outside_loop#0 <-> loop_join_ref#0: MayAlias

func.func @test_rb_do_loop_iter_carry_ref() {
  %lb = arith.constant 1 : index
  %ub = arith.constant 2 : index
  %st = arith.constant 1 : index
  %a_carry = fir.alloca f32 {uniq_name = "_QFEcarry"}
  %d_carry = fir.declare %a_carry {uniq_name = "_QFEcarry", test.ptr = "carry_init"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %a_out = fir.alloca f32 {uniq_name = "_QFEoutside"}
  %d_out = fir.declare %a_out {uniq_name = "_QFEoutside", test.ptr = "outside_loop"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %loop_res = fir.do_loop %iv = %lb to %ub step %st iter_args(%carry = %d_carry) -> (!fir.ref<f32>) {
    fir.result %carry : !fir.ref<f32>
  }
  %join = fir.convert %loop_res {test.ptr = "loop_join_ref"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}
