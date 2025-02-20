// Check aliasing with the address *in* (not *of*) a local (fir.alloca) pointer
// variable.
//
// Throughout this test, the ".fir" suffix on symbols indicates a version of the
// MLIR after convert-hlfir-to-fir.  We would like alias analysis results to be
// the same in both versions.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading  \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// subroutine test(p1, arr, t_arr, alloc, t_alloc, t, v)
//   real, pointer :: p1
//   real :: arr(:)
//   real, target :: t_arr(:)
//   real, allocatable :: alloc
//   real, allocatable, target :: t_alloc
//   real, target :: t
//   real :: v
//   real, pointer :: p0
// end subroutine test

// check when fircg.ext_rebox and fircg.ext_declare are in the path of tracing the source
// CHECK-LABEL: Testing : "_QPtest.fir"
// CHECK-DAG: p0.tgt.fir#0 <-> arr(1).fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_arr(1).fir#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> alloc.tgt.fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_alloc.tgt.fir#0: MayAlias
// CHECK-DAG: alloc.fir#0 <-> alloc.tgt.fir#0: NoAlias

func.func @_QPtest.fir(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "arr"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "t_arr", fir.target}, %arg3: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "alloc"}, %arg4: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "t_alloc", fir.target}, %arg5: !fir.ref<f32> {fir.bindc_name = "t", fir.target}, %arg6: !fir.ref<f32> {fir.bindc_name = "v"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fircg.ext_declare %arg3 dummy_scope %0 {test.ptr = "alloc.fir", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtestEarr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %3 = fircg.ext_rebox %2 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %4 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QFtestEp0"}
  %5 = fircg.ext_declare %4 {test.ptr = "p0.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.declare %arg0 dummy_scope %0 {test.ptr = "p1.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %7 = fir.declare %arg5 dummy_scope %0 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %8 = fir.declare %arg4 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtestEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %9 = fir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_arr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %10 = fircg.ext_rebox %9 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.declare %arg6 dummy_scope %0 {test.ptr = "v.fir", uniq_name = "_QFtestEv"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %12 = fir.load %5 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %13 = fir.box_addr %12 {test.ptr = "p0.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %14 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %15 = fir.box_addr %14 {test.ptr = "p1.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c1 = arith.constant 1 : index
  %16 = fir.array_coor %3 %c1 {test.ptr="arr(1).fir"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %c1_0 = arith.constant 1 : index
  %17 = fir.array_coor %10 %c1_0 {test.ptr="t_arr(1).fir"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %18 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %19 = fir.box_addr %18 {test.ptr = "alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %20 = fir.load %8 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %21 = fir.box_addr %20 {test.ptr = "t_alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

// -----
// CHECK-LABEL: Testing : "_QFPtest3"

// module pointers
//   real, pointer :: p
// end module
//
// program main
//   use pointers
//   real, target :: var1 = 1, var2 =2
//   p => var1
//
//   call test3(p)
//
// contains
//   subroutine test3(p1)
//     real, pointer :: p1
//     p1 => var2
//     print *, p
//   end subroutine
// end

// check when there are fircg.ext_embox in the paths
// CHECK-DAG: p#0 <-> box.addr#0: NoAlias
// CHECK-DAG: box.addr#0 <-> func.region0#0: NoAlias
// CHECK-DAG: var2#0 <-> p#0: NoAlias
// CHECK-DAG: var2#0 <-> box.addr#0: MustAlias
// CHECK-DAG: var2#0 <-> func.region0#1: NoAlias
// CHECK-DAG: box.addr#0 <-> func.region0#1: NoAlias

fir.global @_QMpointersEp : !fir.box<!fir.ptr<f32>> {
  %0 = fir.zero_bits !fir.ptr<f32>
  %1 = fircg.ext_embox %0 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.has_value %1 : !fir.box<!fir.ptr<f32>>
}

fir.global internal @_QFEvar2 target : f32 {
  %cst = arith.constant 2.000000e+00 : f32
  fir.has_value %cst : f32
}

func.func @_QFPtest3(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg1: !fir.ref<f32>) attributes {test.ptr = "func"} {
  %3 = fir.load %arg0 {test.ptr = "arg0.load"}: !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.address_of(@_QFEvar2) {test.ptr = "var2"} : !fir.ref<f32>
  %5 = fir.address_of(@_QMpointersEp) {test.ptr = "p"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fircg.ext_embox %4 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  %13 = fir.box_addr %6 {test.ptr = "box.addr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

