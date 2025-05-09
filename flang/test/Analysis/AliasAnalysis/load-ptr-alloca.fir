// Check aliasing with the address *in* (not *of*) a local (fir.alloca) pointer
// variable.
//
// Throughout this test, the ".fir" suffix on symbols indicates a version of the
// MLIR after convert-hlfir-to-fir.  We would like alias analysis results to be
// the same in both versions.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading  \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// subroutine test(n)
//   integer :: n
//   real, pointer :: p0, p1
//   real :: arr(n)
//   real, target :: t_arr(n)
//   real, allocatable :: alloc
//   real, allocatable, target :: t_alloc
//   real, target :: t
//   real :: v
// end subroutine test

// CHECK-LABEL: Testing : "_QPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: p0.tgt#0 <-> p1.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> p0.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> p1.tgt#0: MayAlias
// CHECK-DAG: v#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: v#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> p1.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> p0.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> p1.tgt.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: v.fir#0 <-> p1.tgt.fir#0: NoAlias

// The address in a pointer cannot alias the address of a pointer.
// CHECK-DAG: p0#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: p0#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p1#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: p1#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p0.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: p0.fir#0 <-> p1.tgt.fir#0: NoAlias
// CHECK-DAG: p1.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: p1.fir#0 <-> p1.tgt.fir#0: NoAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see the address of arr(1) as an address that
// was loaded from a pointer and that could alias something.  However, t_arr is
// a target.
// CHECK-DAG: p0.tgt#0 <-> arr(1)#0: NoAlias
// CHECK-DAG: p0.tgt#0 <-> t_arr(1)#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> arr(1).fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_arr(1).fir#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_alloc is a target.
// CHECK-DAG: p0.tgt#0 <-> alloc.tgt#0: NoAlias
// CHECK-DAG: p0.tgt#0 <-> t_alloc.tgt#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> alloc.tgt.fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_alloc.tgt.fir#0: MayAlias

// The address in an allocatable cannot alias the address of that allocatable.
// CHECK-DAG: alloc#0 <-> alloc.tgt#0: NoAlias
// CHECK-DAG: alloc.fir#0 <-> alloc.tgt.fir#0: NoAlias

func.func @_QPtest(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "alloc", uniq_name = "_QFtestEalloc"}
  %1:2 = hlfir.declare %0 {test.ptr="alloc", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %2:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QFtestEp0"}
  %4:2 = hlfir.declare %3 {test.ptr="p0", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %5 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p1", uniq_name = "_QFtestEp1"}
  %6:2 = hlfir.declare %5 {test.ptr="p1", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %7 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %8:2 = hlfir.declare %7 {test.ptr="t", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %9 = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "t_alloc", fir.target, uniq_name = "_QFtestEt_alloc"}
  %10:2 = hlfir.declare %9 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtestEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %11 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %12:2 = hlfir.declare %11 {test.ptr="v", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %13 = fir.load %2#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> i64
  %15 = fir.convert %14 : (i64) -> index
  %c0 = arith.constant 0 : index
  %16 = arith.cmpi sgt, %15, %c0 : index
  %17 = arith.select %16, %15, %c0 : index
  %18 = fir.alloca !fir.array<?xf32>, %17 {bindc_name = "arr", uniq_name = "_QFtestEarr"}
  %19 = fir.shape %17 : (index) -> !fir.shape<1>
  %20:2 = hlfir.declare %18(%19) {uniq_name = "_QFtestEarr"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %21 = fir.load %2#0 : !fir.ref<i32>
  %22 = fir.convert %21 : (i32) -> i64
  %23 = fir.convert %22 : (i64) -> index
  %c0_0 = arith.constant 0 : index
  %24 = arith.cmpi sgt, %23, %c0_0 : index
  %25 = arith.select %24, %23, %c0_0 : index
  %26 = fir.alloca !fir.array<?xf32>, %25 {bindc_name = "t_arr", fir.target, uniq_name = "_QFtestEt_arr"}
  %27 = fir.shape %25 : (index) -> !fir.shape<1>
  %28:2 = hlfir.declare %26(%27) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_arr"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %29 = fir.load %4#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %30 = fir.box_addr %29 {test.ptr="p0.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %31 = fir.load %6#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %32 = fir.box_addr %31 {test.ptr="p1.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c1 = arith.constant 1 : index
  %33 = hlfir.designate %20#0 (%c1) {test.ptr="arr(1)"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %c1_1 = arith.constant 1 : index
  %34 = hlfir.designate %28#0 (%c1_1) {test.ptr="t_arr(1)"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %35 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %36 = fir.box_addr %35 {test.ptr="alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %37 = fir.load %10#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %38 = fir.box_addr %37 {test.ptr="t_alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

func.func @_QPtest.fir(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "alloc", uniq_name = "_QFtestEalloc"}
  %1 = fir.declare %0 {test.ptr = "alloc.fir", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.declare %arg0 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QFtestEp0"}
  %4 = fir.declare %3 {test.ptr = "p0.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p1", uniq_name = "_QFtestEp1"}
  %6 = fir.declare %5 {test.ptr = "p1.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %7 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %8 = fir.declare %7 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "t_alloc", fir.target, uniq_name = "_QFtestEt_alloc"}
  %10 = fir.declare %9 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtestEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %11 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %12 = fir.declare %11 {test.ptr = "v.fir", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %13 = fir.load %2 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> i64
  %15 = fir.convert %14 : (i64) -> index
  %c0 = arith.constant 0 : index
  %16 = arith.cmpi sgt, %15, %c0 : index
  %17 = arith.select %16, %15, %c0 : index
  %18 = fir.alloca !fir.array<?xf32>, %17 {bindc_name = "arr", uniq_name = "_QFtestEarr"}
  %19 = fir.shape %17 : (index) -> !fir.shape<1>
  %20 = fir.declare %18(%19) {uniq_name = "_QFtestEarr"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  %21 = fir.embox %20(%19) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %22 = fir.load %2 : !fir.ref<i32>
  %23 = fir.convert %22 : (i32) -> i64
  %24 = fir.convert %23 : (i64) -> index
  %c0_0 = arith.constant 0 : index
  %25 = arith.cmpi sgt, %24, %c0_0 : index
  %26 = arith.select %25, %24, %c0_0 : index
  %27 = fir.alloca !fir.array<?xf32>, %26 {bindc_name = "t_arr", fir.target, uniq_name = "_QFtestEt_arr"}
  %28 = fir.shape %26 : (index) -> !fir.shape<1>
  %29 = fir.declare %27(%28) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_arr"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  %30 = fir.embox %29(%28) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %31 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %32 = fir.box_addr %31 {test.ptr = "p0.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %33 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %34 = fir.box_addr %33 {test.ptr = "p1.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c1 = arith.constant 1 : index
  %35 = fir.array_coor %20(%19) %c1 {test.ptr="arr(1).fir"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %c1_1 = arith.constant 1 : index
  %36 = fir.array_coor %29(%28) %c1_1 {test.ptr="t_arr(1).fir"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %37 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %38 = fir.box_addr %37 {test.ptr = "alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %39 = fir.load %10 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %40 = fir.box_addr %39 {test.ptr = "t_alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

// -----

// Repeat above test except compare the local pointer against dummy args instead
// of other locals.

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

// CHECK-LABEL: Testing : "_QPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: p0.tgt#0 <-> p1.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> p0.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> p1.tgt#0: MayAlias
// CHECK-DAG: v#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: v#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> p1.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> p0.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> p1.tgt.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: v.fir#0 <-> p1.tgt.fir#0: NoAlias

// The address in a pointer cannot alias the address of a pointer.
// CHECK-DAG: p0#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: p0#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p1#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: p1#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p0.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: p0.fir#0 <-> p1.tgt.fir#0: NoAlias
// CHECK-DAG: p1.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: p1.fir#0 <-> p1.tgt.fir#0: NoAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see the address of arr(1) as an address that
// was loaded from a pointer and that could alias something.  However, t_arr is
// a target.
// CHECK-DAG: p0.tgt#0 <-> arr(1)#0: NoAlias
// CHECK-DAG: p0.tgt#0 <-> t_arr(1)#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> arr(1).fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_arr(1).fir#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_alloc is a target.
// CHECK-DAG: p0.tgt#0 <-> alloc.tgt#0: NoAlias
// CHECK-DAG: p0.tgt#0 <-> t_alloc.tgt#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> alloc.tgt.fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_alloc.tgt.fir#0: MayAlias

// The address in an allocatable cannot alias the address of that allocatable.
// CHECK-DAG: alloc#0 <-> alloc.tgt#0: NoAlias
// CHECK-DAG: alloc.fir#0 <-> alloc.tgt.fir#0: NoAlias

func.func @_QPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "arr"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "t_arr", fir.target}, %arg3: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "alloc"}, %arg4: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "t_alloc", fir.target}, %arg5: !fir.ref<f32> {fir.bindc_name = "t", fir.target}, %arg6: !fir.ref<f32> {fir.bindc_name = "v"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg3 dummy_scope %0 {test.ptr="alloc", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtestEarr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QFtestEp0"}
  %4:2 = hlfir.declare %3 {test.ptr="p0", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %5:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="p1", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %6:2 = hlfir.declare %arg5 dummy_scope %0 {test.ptr="t", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %7:2 = hlfir.declare %arg4 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtestEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %8:2 = hlfir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_arr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %9:2 = hlfir.declare %arg6 dummy_scope %0 {test.ptr="v", uniq_name = "_QFtestEv"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %10 = fir.load %4#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.box_addr %10 {test.ptr="p0.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %12 = fir.load %5#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %13 = fir.box_addr %12 {test.ptr="p1.tgt"}  : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c1 = arith.constant 1 : index
  %14 = hlfir.designate %2#0 (%c1) {test.ptr="arr(1)"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %c1_0 = arith.constant 1 : index
  %15 = hlfir.designate %8#0 (%c1_0) {test.ptr="t_arr(1)"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %16 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %17 = fir.box_addr %16 {test.ptr="alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %18 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %19 = fir.box_addr %18 {test.ptr="t_alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

func.func @_QPtest.fir(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "arr"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "t_arr", fir.target}, %arg3: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "alloc"}, %arg4: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "t_alloc", fir.target}, %arg5: !fir.ref<f32> {fir.bindc_name = "t", fir.target}, %arg6: !fir.ref<f32> {fir.bindc_name = "v"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg3 dummy_scope %0 {test.ptr = "alloc.fir", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtestEarr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %3 = fir.rebox %2 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %4 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QFtestEp0"}
  %5 = fir.declare %4 {test.ptr = "p0.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.declare %arg0 dummy_scope %0 {test.ptr = "p1.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %7 = fir.declare %arg5 dummy_scope %0 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %8 = fir.declare %arg4 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtestEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %9 = fir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_arr"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.rebox %9 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
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

// Repeat above test except compare the local pointer against globals.

// module m
//   real, pointer :: p1
//   real :: arr(2)
//   real, target :: t_arr(2)
//   real, allocatable :: alloc
//   real, allocatable, target :: t_alloc
//   real, target :: t
//   real :: v
// contains
//   subroutine test()
//     real, pointer :: p0
//   end subroutine test
// end module m

// CHECK-LABEL: Testing : "_QMmPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: p0.tgt#0 <-> p1.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> p0.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> p1.tgt#0: MayAlias
// CHECK-DAG: v#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: v#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> p1.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> p0.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> p1.tgt.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: v.fir#0 <-> p1.tgt.fir#0: NoAlias

// The address in a pointer cannot alias the address of a pointer.
// CHECK-DAG: p0#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: p0#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p1#0 <-> p0.tgt#0: NoAlias
// CHECK-DAG: p1#0 <-> p1.tgt#0: NoAlias
// CHECK-DAG: p0.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: p0.fir#0 <-> p1.tgt.fir#0: NoAlias
// CHECK-DAG: p1.fir#0 <-> p0.tgt.fir#0: NoAlias
// CHECK-DAG: p1.fir#0 <-> p1.tgt.fir#0: NoAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see the address of arr(1) as an address that
// was loaded from a pointer and that could alias something.  However, t_arr is
// a target.
// CHECK-DAG: p0.tgt#0 <-> arr(1)#0: NoAlias
// CHECK-DAG: p0.tgt#0 <-> t_arr(1)#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> arr(1).fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_arr(1).fir#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_alloc is a target.
// CHECK-DAG: p0.tgt#0 <-> alloc.tgt#0: NoAlias
// CHECK-DAG: p0.tgt#0 <-> t_alloc.tgt#0: MayAlias
// CHECK-DAG: p0.tgt.fir#0 <-> alloc.tgt.fir#0: NoAlias
// CHECK-DAG: p0.tgt.fir#0 <-> t_alloc.tgt.fir#0: MayAlias

// The address in an allocatable cannot alias the address of that allocatable.
// CHECK-DAG: alloc#0 <-> alloc.tgt#0: NoAlias
// CHECK-DAG: alloc.fir#0 <-> alloc.tgt.fir#0: NoAlias

func.func @_QMmPtest() {
  %0 = fir.address_of(@_QMmEalloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %1:2 = hlfir.declare %0 {test.ptr="alloc", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %2 = fir.address_of(@_QMmEarr) : !fir.ref<!fir.array<2xf32>>
  %c2 = arith.constant 2 : index
  %3 = fir.shape %c2 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %2(%3) {uniq_name = "_QMmEarr"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
  %5 = fir.address_of(@_QMmEp1) : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6:2 = hlfir.declare %5 {test.ptr="p1", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %7 = fir.address_of(@_QMmEt) : !fir.ref<f32>
  %8:2 = hlfir.declare %7 {test.ptr="t", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %9 = fir.address_of(@_QMmEt_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %10:2 = hlfir.declare %9 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %11 = fir.address_of(@_QMmEt_arr) : !fir.ref<!fir.array<2xf32>>
  %c2_0 = arith.constant 2 : index
  %12 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %13:2 = hlfir.declare %11(%12) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEt_arr"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
  %14 = fir.address_of(@_QMmEv) : !fir.ref<f32>
  %15:2 = hlfir.declare %14 {test.ptr="v", uniq_name = "_QMmEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %16 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QMmFtestEp0"}
  %17:2 = hlfir.declare %16 {test.ptr="p0", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %18 = fir.load %17#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %19 = fir.box_addr %18 {test.ptr="p0.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %20 = fir.load %6#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %21 = fir.box_addr %20 {test.ptr="p1.tgt"}  : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c1 = arith.constant 1 : index
  %22 = hlfir.designate %4#0 (%c1) {test.ptr="arr(1)"} : (!fir.ref<!fir.array<2xf32>>, index) -> !fir.ref<f32>
  %c1_1 = arith.constant 1 : index
  %23 = hlfir.designate %13#0 (%c1_1) {test.ptr="t_arr(1)"} : (!fir.ref<!fir.array<2xf32>>, index) -> !fir.ref<f32>
  %24 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %25 = fir.box_addr %24 {test.ptr="alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %26 = fir.load %10#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %27 = fir.box_addr %26 {test.ptr="t_alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

func.func @_QMmPtest.fir() {
  %0 = fir.address_of(@_QMmEalloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %1 = fir.declare %0 {test.ptr = "alloc.fir", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.address_of(@_QMmEarr) : !fir.ref<!fir.array<2xf32>>
  %c2 = arith.constant 2 : index
  %3 = fir.shape %c2 : (index) -> !fir.shape<1>
  %4 = fir.declare %2(%3) {uniq_name = "_QMmEarr"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xf32>>
  %5 = fir.address_of(@_QMmEp1) : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.declare %5 {test.ptr = "p1.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmEp1"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %7 = fir.address_of(@_QMmEt) : !fir.ref<f32>
  %8 = fir.declare %7 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.address_of(@_QMmEt_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %10 = fir.declare %9 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %11 = fir.address_of(@_QMmEt_arr) : !fir.ref<!fir.array<2xf32>>
  %c2_0 = arith.constant 2 : index
  %12 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %13 = fir.declare %11(%12) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEt_arr"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xf32>>
  %14 = fir.address_of(@_QMmEv) : !fir.ref<f32>
  %15 = fir.declare %14 {test.ptr = "v.fir", uniq_name = "_QMmEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %16 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p0", uniq_name = "_QMmFtestEp0"}
  %17 = fir.declare %16 {test.ptr = "p0.fir", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFtestEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %18 = fir.load %17 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %19 = fir.box_addr %18 {test.ptr = "p0.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %20 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %21 = fir.box_addr %20 {test.ptr = "p1.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c1 = arith.constant 1 : index
  %22 = fir.array_coor %4(%3) %c1 {test.ptr="arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %c1_1 = arith.constant 1 : index
  %23 = fir.array_coor %13(%12) %c1_1 {test.ptr="t_arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %24 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %25 = fir.box_addr %24 {test.ptr = "alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %26 = fir.load %10 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %27 = fir.box_addr %26 {test.ptr = "t_alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}
