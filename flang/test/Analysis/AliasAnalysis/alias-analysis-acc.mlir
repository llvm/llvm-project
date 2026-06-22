// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// -----

// Two acc.copyin results from distinct host allocas do not alias.
// CHECK-LABEL: Testing : "testBothOutsideCopyinDistinctHosts"
// CHECK-DAG: cin_a#0 <-> cin_b#0: NoAlias

func.func @testBothOutsideCopyinDistinctHosts() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a", test.ptr = "cin_a"}
  %cb = acc.copyin varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b", test.ptr = "cin_b"}
  return
}

// -----

// Two acc.copyin results from dummy arguments that are Fortran TARGET variables:
// they may alias.
// CHECK-LABEL: Testing : "testBothOutsideCopyinTargetDummyArgsMayAlias"
// CHECK-DAG: arg_cp_a#0 <-> arg_cp_b#0: MayAlias

func.func @testBothOutsideCopyinTargetDummyArgsMayAlias(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %ds = fir.dummy_scope : !fir.dscope
  %dx = fir.declare %arg0 dummy_scope %ds arg 1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEex"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %dy = fir.declare %arg1 dummy_scope %ds arg 2 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEey"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %cx = acc.copyin varPtr(%dx : !fir.ref<f32>) -> !fir.ref<f32> {name = "x", test.ptr = "arg_cp_a"}
  %cy = acc.copyin varPtr(%dy : !fir.ref<f32>) -> !fir.ref<f32> {name = "y", test.ptr = "arg_cp_b"}
  return
}

// -----

// Two acc.copyin results mapping the same host ref must alias.
// CHECK-LABEL: Testing : "testBothOutsideCopyinSameHostMustAlias"
// CHECK-DAG: out_must_a#0 <-> out_must_b#0: MustAlias

func.func @testBothOutsideCopyinSameHostMustAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca1 = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a", test.ptr = "out_must_a"}
  %ca2 = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a", test.ptr = "out_must_b"}
  return
}

// -----

// CHECK-LABEL: Testing : "testBothOutsideCreateDistinctHosts"
// CHECK-DAG: crt_a#0 <-> crt_b#0: NoAlias

func.func @testBothOutsideCreateDistinctHosts() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ta = acc.create varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a", test.ptr = "crt_a"}
  %tb = acc.create varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b", test.ptr = "crt_b"}
  return
}

// -----

// Same distinct-host copyins as above, but threaded through acc.compute_region
// block arguments.
// CHECK-LABEL: Testing : "testComputeRegionCopyinDistinctHostsInsideConvert"
// CHECK-DAG: cr_dist_a#0 <-> cr_dist_b#0: NoAlias

func.func @testComputeRegionCopyinDistinctHostsInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %cb = acc.copyin varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
  acc.compute_region ins(%arg0 = %ca, %arg1 = %cb) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_dist_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_dist_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.kernels"}
  return
}

// -----

// CHECK-LABEL: Testing : "testComputeRegionCreateDistinctHostsInsideConvert"
// CHECK-DAG: cr_crt_a#0 <-> cr_crt_b#0: NoAlias

func.func @testComputeRegionCreateDistinctHostsInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ta = acc.create varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %tb = acc.create varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
  acc.compute_region ins(%arg0 = %ta, %arg1 = %tb) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_crt_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_crt_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.kernels"}
  return
}

// -----

// Same TARGET dummy copyins as testBothOutsideCopyinTargetDummyArgsMayAlias,
// through acc.compute_region block args.
// CHECK-LABEL: Testing : "testComputeRegionCopyinTargetDummiesMayAliasInsideConvert"
// CHECK-DAG: cr_tgt_a#0 <-> cr_tgt_b#0: MayAlias

func.func @testComputeRegionCopyinTargetDummiesMayAliasInsideConvert(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %ds = fir.dummy_scope : !fir.dscope
  %dx = fir.declare %arg0 dummy_scope %ds arg 1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEex"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %dy = fir.declare %arg1 dummy_scope %ds arg 2 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEey"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %cx = acc.copyin varPtr(%dx : !fir.ref<f32>) -> !fir.ref<f32> {name = "x"}
  %cy = acc.copyin varPtr(%dy : !fir.ref<f32>) -> !fir.ref<f32> {name = "y"}
  acc.compute_region ins(%cr0 = %cx, %cr1 = %cy) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %cr0 {test.ptr = "cr_tgt_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %cr1 {test.ptr = "cr_tgt_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

// Single host copyin wired twice through arguments; both block args alias the
// same mapped host variable.
// CHECK-LABEL: Testing : "testComputeRegionCopyinSameHostMustAliasInsideConvert"
// CHECK-DAG: cr_must_a#0 <-> cr_must_b#0: MustAlias

func.func @testComputeRegionCopyinSameHostMustAliasInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %ca, %arg1 = %ca) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_must_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_must_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.kernels"}
  return
}

// -----

// Distinct-host copyins passed as acc.kernels dataOperands.
// CHECK-LABEL: Testing : "testKernelsCopyinDistinctHostsInsideConvert"
// CHECK-DAG: kern_dist_a#0 <-> kern_dist_b#0: NoAlias

func.func @testKernelsCopyinDistinctHostsInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %cb = acc.copyin varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
  acc.kernels dataOperands(%ca, %cb : !fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %ca {test.ptr = "kern_dist_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %cb {test.ptr = "kern_dist_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// CHECK-LABEL: Testing : "testKernelsCreateDistinctHostsInsideConvert"
// CHECK-DAG: kern_crt_a#0 <-> kern_crt_b#0: NoAlias

func.func @testKernelsCreateDistinctHostsInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ta = acc.create varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %tb = acc.create varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
  acc.kernels dataOperands(%ta, %tb : !fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %ta {test.ptr = "kern_crt_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %tb {test.ptr = "kern_crt_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// TARGET dummy copyins as acc.kernels dataOperands.
// CHECK-LABEL: Testing : "testKernelsCopyinTargetDummiesMayAliasInsideConvert"
// CHECK-DAG: kern_tgt_a#0 <-> kern_tgt_b#0: MayAlias

func.func @testKernelsCopyinTargetDummiesMayAliasInsideConvert(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %ds = fir.dummy_scope : !fir.dscope
  %dx = fir.declare %arg0 dummy_scope %ds arg 1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEex"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %dy = fir.declare %arg1 dummy_scope %ds arg 2 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEey"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %cx = acc.copyin varPtr(%dx : !fir.ref<f32>) -> !fir.ref<f32> {name = "x"}
  %cy = acc.copyin varPtr(%dy : !fir.ref<f32>) -> !fir.ref<f32> {name = "y"}
  acc.kernels dataOperands(%cx, %cy : !fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %cx {test.ptr = "kern_tgt_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %cy {test.ptr = "kern_tgt_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// Same copyin value listed twice in dataOperands; both converts must alias.
// CHECK-LABEL: Testing : "testKernelsCopyinSameHostMustAliasInsideConvert"
// CHECK-DAG: kern_must_a#0 <-> kern_must_b#0: MustAlias

func.func @testKernelsCopyinSameHostMustAliasInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.kernels dataOperands(%ca : !fir.ref<f32>) {
    %va = fir.convert %ca {test.ptr = "kern_must_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %ca {test.ptr = "kern_must_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// acc.compute_region: both queried values are inside the region; test.ptr is on
// fir.convert of each captured private (traces block operands without tagging
// the region op).
// CHECK-LABEL: Testing : "testComputeRegionPrivateInsideConvert"
// CHECK-DAG: cr_priv_a#0 <-> cr_priv_b#0: NoAlias

func.func @testComputeRegionPrivateInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %pa = acc.private varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pb = acc.private varPtr(%db : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
  acc.compute_region ins(%arg0 = %pa, %arg1 = %pb) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_priv_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_priv_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

// Same host ref as both acc.copyin and acc.private operands (two copyins would
// must-alias); private-like mapping is a distinct allocation vs copyin.
// CHECK-LABEL: Testing : "testComputeRegionCopyinVsPrivateSameHostNoAlias"
// CHECK-DAG: cr_mix_cp#0 <-> cr_mix_pr#0: NoAlias

func.func @testComputeRegionCopyinVsPrivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pp = acc.private varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %ca, %arg1 = %pp) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_mix_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_mix_pr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

// CHECK-LABEL: Testing : "testComputeRegionCreateVsPrivateSameHostNoAlias"
// CHECK-DAG: cr_mix_cr#0 <-> cr_mix_pr2#0: NoAlias

func.func @testComputeRegionCreateVsPrivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ta = acc.create varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pp = acc.private varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %ta, %arg1 = %pp) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_mix_cr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_mix_pr2"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

// Same host: acc.copyin vs acc.firstprivate.
// CHECK-LABEL: Testing : "testComputeRegionCopyinVsFirstprivateSameHostNoAlias"
// CHECK-DAG: cr_fp_cp#0 <-> cr_fp_pr#0: NoAlias

func.func @testComputeRegionCopyinVsFirstprivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pf = acc.firstprivate varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %ca, %arg1 = %pf) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_fp_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_fp_pr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

// Same host: acc.copyin vs acc.firstprivate_map.
// CHECK-LABEL: Testing : "testComputeRegionCopyinVsFirstprivateMapSameHostNoAlias"
// CHECK-DAG: cr_fpm_cp#0 <-> cr_fpm_fm#0: NoAlias

func.func @testComputeRegionCopyinVsFirstprivateMapSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %fm = acc.firstprivate_map varPtr(%da : !fir.ref<f32>) varType(f32) -> !fir.ref<f32> {name = "a"}
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %ca, %arg1 = %fm) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_fpm_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_fpm_fm"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.kernels"}
  return
}

// -----

// acc.kernels: same host as copyin and acc.private dataOperands.
// CHECK-LABEL: Testing : "testKernelsCopyinVsPrivateSameHostNoAlias"
// CHECK-DAG: k_mix_cp#0 <-> k_mix_pr#0: NoAlias

func.func @testKernelsCopyinVsPrivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pp = acc.private varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.kernels dataOperands(%ca : !fir.ref<f32>) private(%pp : !fir.ref<f32>) {
    %va = fir.convert %ca {test.ptr = "k_mix_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %pp {test.ptr = "k_mix_pr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// CHECK-LABEL: Testing : "testKernelsCreateVsPrivateSameHostNoAlias"
// CHECK-DAG: k_mix_cr#0 <-> k_mix_pr2#0: NoAlias

func.func @testKernelsCreateVsPrivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ta = acc.create varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pp = acc.private varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.kernels dataOperands(%ta : !fir.ref<f32>) private(%pp : !fir.ref<f32>) {
    %va = fir.convert %ta {test.ptr = "k_mix_cr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %pp {test.ptr = "k_mix_pr2"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// acc.kernels: acc.copyin vs acc.firstprivate.
// CHECK-LABEL: Testing : "testKernelsCopyinVsFirstprivateSameHostNoAlias"
// CHECK-DAG: k_fp_cp#0 <-> k_fp_pr#0: NoAlias

func.func @testKernelsCopyinVsFirstprivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  %pf = acc.firstprivate varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.kernels dataOperands(%ca : !fir.ref<f32>) firstprivate(%pf : !fir.ref<f32>) {
    %va = fir.convert %ca {test.ptr = "k_fp_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %pf {test.ptr = "k_fp_pr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// acc.kernels: acc.copyin vs acc.firstprivate_map.
// CHECK-LABEL: Testing : "testKernelsCopyinVsFirstprivateMapSameHostNoAlias"
// CHECK-DAG: k_fpm_cp#0 <-> k_fpm_fm#0: NoAlias

func.func @testKernelsCopyinVsFirstprivateMapSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %fm = acc.firstprivate_map varPtr(%da : !fir.ref<f32>) varType(f32) -> !fir.ref<f32> {name = "a"}
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.kernels dataOperands(%ca : !fir.ref<f32>) firstprivate(%fm : !fir.ref<f32>) {
    %va = fir.convert %ca {test.ptr = "k_fpm_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %fm {test.ptr = "k_fpm_fm"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

// acc.private inside acc.compute_region; ins carries acc.create from the host ref.
// CHECK-LABEL: Testing : "testComputeRegionPrivateOpInsideVsInsCreateNoAlias"
// CHECK-DAG: cr_body_pr#0 <-> cr_body_cr#0: NoAlias

func.func @testComputeRegionPrivateOpInsideVsInsCreateNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %tc = acc.create varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %tc) : (!fir.ref<f32>) {
    %pv = acc.private varPtr(%arg0 : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
    %vb = fir.convert %pv {test.ptr = "cr_body_pr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vc = fir.convert %arg0 {test.ptr = "cr_body_cr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

// acc.private inside acc.kernels on the dataOperand copyin of the host ref.
// CHECK-LABEL: Testing : "testKernelsPrivateOpInsideVsDataCopyinNoAlias"
// CHECK-DAG: k_body_pr#0 <-> k_body_cp#0: NoAlias

func.func @testKernelsPrivateOpInsideVsDataCopyinNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ca = acc.copyin varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.kernels dataOperands(%ca : !fir.ref<f32>) {
    %pv = acc.private varPtr(%ca : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
    %vb = fir.convert %pv {test.ptr = "k_body_pr"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vc = fir.convert %ca {test.ptr = "k_body_cp"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// -----

acc.reduction.recipe @red_f32_aa : !fir.ref<f32> reduction_operator <add> init {
^bb0(%arg0: !fir.ref<f32>):
  %init = fir.alloca f32
  acc.yield %init : !fir.ref<f32>
} combiner {
^bb0(%lhs: !fir.ref<f32>, %rhs: !fir.ref<f32>):
  %lv = fir.load %lhs : !fir.ref<f32>
  %rv = fir.load %rhs : !fir.ref<f32>
  %s = arith.addf %lv, %rv : f32
  %out = fir.alloca f32
  fir.store %s to %out : !fir.ref<f32>
  acc.yield %out : !fir.ref<f32>
}

// CHECK-LABEL: Testing : "testBothOutsideReductionDistinctHosts"
// CHECK-DAG: red_a#0 <-> red_b#0: NoAlias

func.func @testBothOutsideReductionDistinctHosts() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ra = acc.reduction varPtr(%da : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "a", test.ptr = "red_a"}
  %rb = acc.reduction varPtr(%db : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "b", test.ptr = "red_b"}
  return
}

// CHECK-LABEL: Testing : "testComputeRegionReductionDistinctHostsInsideConvert"
// CHECK-DAG: cr_red_a#0 <-> cr_red_b#0: NoAlias

func.func @testComputeRegionReductionDistinctHostsInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ra = acc.reduction varPtr(%da : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "a"}
  %rb = acc.reduction varPtr(%db : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "b"}
  acc.compute_region ins(%arg0 = %ra, %arg1 = %rb) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_red_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_red_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: Testing : "testKernelsReductionDistinctHostsInsideConvert"
// CHECK-DAG: kern_red_a#0 <-> kern_red_b#0: NoAlias

func.func @testKernelsReductionDistinctHostsInsideConvert() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %b = fir.alloca f32 {uniq_name = "_QFEb"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %db = fir.declare %b {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ra = acc.reduction varPtr(%da : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "a"}
  %rb = acc.reduction varPtr(%db : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "b"}
  acc.kernels reduction(%ra, %rb : !fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %ra {test.ptr = "kern_red_a"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %rb {test.ptr = "kern_red_b"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.terminator
  }
  return
}

// CHECK-LABEL: Testing : "testComputeRegionReductionVsPrivateSameHostNoAlias"
// CHECK-DAG: cr_mix_rd#0 <-> cr_mix_pr3#0: NoAlias

func.func @testComputeRegionReductionVsPrivateSameHostNoAlias() {
  %a = fir.alloca f32 {uniq_name = "_QFEa"}
  %da = fir.declare %a {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %ra = acc.reduction varPtr(%da : !fir.ref<f32>) recipe(@red_f32_aa) -> !fir.ref<f32> {name = "a"}
  %pp = acc.private varPtr(%da : !fir.ref<f32>) -> !fir.ref<f32> {name = "a"}
  acc.compute_region ins(%arg0 = %ra, %arg1 = %pp) : (!fir.ref<f32>, !fir.ref<f32>) {
    %va = fir.convert %arg0 {test.ptr = "cr_mix_rd"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %vb = fir.convert %arg1 {test.ptr = "cr_mix_pr3"} : (!fir.ref<f32>) -> !fir.ref<f32>
    acc.yield
  } {origin = "acc.kernels"}
  return
}

// -----

// Allocas inside acc.parallel.
// CHECK-LABEL: Testing : "testBothInsideParallel"
// CHECK-DAG: par_a#0 <-> par_b#0: NoAlias

func.func @testBothInsideParallel() {
  acc.parallel {
    %a = fir.alloca f32 {uniq_name = "_QFEa", test.ptr = "par_a"}
    %b = fir.alloca f32 {uniq_name = "_QFEb", test.ptr = "par_b"}
    acc.yield
  }
  return
}

// -----

// Allocas inside acc.kernels.
// CHECK-LABEL: Testing : "testBothInsideKernels"
// CHECK-DAG: kern_a#0 <-> kern_b#0: NoAlias

func.func @testBothInsideKernels() {
  acc.kernels {
    %a = fir.alloca f32 {uniq_name = "_QFEa", test.ptr = "kern_a"}
    %b = fir.alloca f32 {uniq_name = "_QFEb", test.ptr = "kern_b"}
    acc.terminator
  }
  return
}

// -----

// Test passing through acc.compute_region boundary inside acc routine.
// The two dummy arguments cannot alias:
// CHECK-LABEL: Testing : "test_acc_routine__0"
// CHECK-DAG: arg_a#0 <-> arg_b#0: NoAlias
func.func @test_acc_routine__0(%arg0: !fir.ref<f32> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "b"}) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_0, <vector>, "test_acc_routine_">, fir.internal_name = "_QPtest_acc_routine"} {
  %0 = acc.par_width {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%arg2 = %0) ins(%arg3 = %arg0, %arg4 = %arg1) : (!fir.ref<f32>, !fir.ref<f32>) {
    %1 = fir.dummy_scope : !fir.dscope
    %2 = fir.declare %arg3 dummy_scope %1 arg 1 {uniq_name = "_QFtest_acc_routineEa", test.ptr = "arg_a"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
    %3 = fir.declare %arg4 dummy_scope %1 arg 2 {uniq_name = "_QFtest_acc_routineEb", test.ptr = "arg_b"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
    %4 = fir.load %3 : !fir.ref<f32>
    fir.store %4 to %2 : !fir.ref<f32>
    acc.yield
  } {origin = "acc.routine"}
  return
}

// -----

// Test that a fir.load of a box descriptor through an acc.compute_region
// block argument still resolves to its dummy-argument source. Without the
// pass-through, the recursive getSource() on the memref of the load would
// return SourceKind::Unknown for the block argument, forcing the load itself
// to be classified as SourceKind::Indirect and pessimizing alias analysis
// to MayAlias for two unrelated allocatable dummy arguments.
// CHECK-LABEL: Testing : "test_acc_compute_region_box_load"
// CHECK-DAG: load_x#0 <-> load_y#0: NoAlias
func.func @test_acc_compute_region_box_load(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "y"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %dx = fir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %dy = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEy"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %cx = acc.copyin varPtr(%dx : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "x"}
  %cy = acc.copyin varPtr(%dy : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "y"}
  acc.kernel_environment dataOperands(%cx, %cy : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    acc.compute_region ins(%arg2 = %cx, %arg3 = %cy) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
      %lx = fir.load %arg2 {test.ptr = "load_x"} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      %ly = fir.load %arg3 {test.ptr = "load_y"} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      acc.yield
    } {origin = "acc.kernels"}
  }
  return
}
