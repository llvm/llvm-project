// Check aliasing with the address passed via a pointer dummy argument.

// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   --mlir-disable-threading 2>&1 | FileCheck -match-full-lines %s

// subroutine test(p0, p1, arr, t_arr, alloc, t_alloc)
//   real, pointer :: p0, p1
//   real :: arr(:)
//   real, target :: t_arr(:)
//   real, allocatable :: alloc
//   real, allocatable, target :: t_alloc
//   real, target :: t
//   real :: v
//   v = p0
//   v = p1
//   v = arr(1)
//   v = t_arr(1)
//   v = alloc
//   v = t_alloc
// end subroutine test

// CHECK-LABEL: Testing : "_QPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: t.addr#0 <-> p0.tgt_addr#0: MayAlias
// CHECK-DAG: t.addr#0 <-> p1.tgt_addr#0: MayAlias
// CHECK-DAG: v.addr#0 <-> p0.tgt_addr#0: NoAlias
// CHECK-DAG: v.addr#0 <-> p1.tgt_addr#0: NoAlias
// CHECK-DAG: p0.tgt_addr#0 <-> p1.tgt_addr#0: MayAlias

// Determining whether the address *in* a pointer can alias the address *of* a
// pointer is not yet handled.  In the past, when it was the same pointer, that
// relationship was mistakenly determined to be MustAlias.
// CHECK-DAG: p0.tgt_addr#0 <-> func.region0#0: MayAlias
// CHECK-DAG: p0.tgt_addr#0 <-> func.region0#1: MayAlias
// CHECK-DAG: p1.tgt_addr#0 <-> func.region0#0: MayAlias
// CHECK-DAG: p1.tgt_addr#0 <-> func.region0#1: MayAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see arr(1).addr as an address that was loaded
// from a pointer and that could alias something.  However, t_arr is a target.
// CHECK-DAG: p0.tgt_addr#0 <-> arr(1).addr#0: NoAlias
// CHECK-DAG: p0.tgt_addr#0 <-> t_arr(1).addr#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_alloc is a target.
// CHECK-DAG: p0.tgt_addr#0 <-> alloc.tgt_addr#0: NoAlias
// CHECK-DAG: p0.tgt_addr#0 <-> t_alloc.tgt_addr#0: MayAlias

// The address *in* an allocatable cannot alias the address *of* that
// allocatable.
// CHECK-DAG: alloc.addr#0 <-> alloc.tgt_addr#0: NoAlias

func.func @_QPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p0"}, %arg1: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "arr"}, %arg3: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "t_arr", fir.target}, %arg4: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "alloc"}, %arg5: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "t_alloc", fir.target}) attributes {test.ptr="func"} {
  %0:2 = hlfir.declare %arg4 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEalloc", test.ptr="alloc.addr"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEarr"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %3:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %4 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %5:2 = hlfir.declare %4 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt", test.ptr="t.addr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %6:2 = hlfir.declare %arg5 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtestEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %7:2 = hlfir.declare %arg3 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_arr"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %8 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %9:2 = hlfir.declare %8 {uniq_name = "_QFtestEv", test.ptr="v.addr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %10 = fir.load %2#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.box_addr %10 {test.ptr="p0.tgt_addr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %12 = fir.load %11 : !fir.ptr<f32>
  hlfir.assign %12 to %9#0 : f32, !fir.ref<f32>
  %13 = fir.load %3#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %14 = fir.box_addr %13 {test.ptr="p1.tgt_addr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %15 = fir.load %14 : !fir.ptr<f32>
  hlfir.assign %15 to %9#0 : f32, !fir.ref<f32>
  %c1 = arith.constant 1 : index
  %16 = hlfir.designate %1#0 (%c1) {test.ptr="arr(1).addr"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %17 = fir.load %16 : !fir.ref<f32>
  hlfir.assign %17 to %9#0 : f32, !fir.ref<f32>
  %c1_0 = arith.constant 1 : index
  %18 = hlfir.designate %7#0 (%c1_0) {test.ptr="t_arr(1).addr"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %19 = fir.load %18 : !fir.ref<f32>
  hlfir.assign %19 to %9#0 : f32, !fir.ref<f32>
  %20 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %21 = fir.box_addr %20 {test.ptr="alloc.tgt_addr"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %22 = fir.load %21 : !fir.heap<f32>
  hlfir.assign %22 to %9#0 : f32, !fir.ref<f32>
  %23 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %24 = fir.box_addr %23 {test.ptr="t_alloc.tgt_addr"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %25 = fir.load %24 : !fir.heap<f32>
  hlfir.assign %25 to %9#0 : f32, !fir.ref<f32>
  return
}
