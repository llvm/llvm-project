// Check aliasing with the address *in* (not *of*) a pointer component
// (hlfir.designate).
//
// Throughout this test, the ".fir" suffix on symbols indicates a version of the
// MLIR after convert-hlfir-to-fir.  A key difference is that component access
// is via fir.coordinate_of instead of hlfir.designate.  We would like alias
// analysis results to be the same in both versions.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading  \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// module m
//   type :: ty
//     real, pointer :: p0, p1
//     real :: arr(2)
//     real, allocatable :: alloc
//     ! target attribute on components is not supported
//   end type ty
// end module m
// subroutine test()
//   use m
//   real, target :: t
//   real :: v
//   type(ty) :: obj
//   type(ty), target :: t_obj
// end subroutine test

// CHECK-LABEL: Testing : "_QPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%p1.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> obj%p0.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> obj%p1.tgt#0: MayAlias
// CHECK-DAG: v#0 <-> obj%p0.tgt#0: NoAlias
// CHECK-DAG: v#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%p1.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> obj%p0.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> obj%p1.tgt.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> obj%p0.tgt.fir#0: NoAlias
// CHECK-DAG: v.fir#0 <-> obj%p1.tgt.fir#0: NoAlias

// The address in a pointer cannot alias the address of a pointer.
// CHECK-DAG: obj%p0#0 <-> obj%p0.tgt#0: NoAlias
// CHECK-DAG: obj%p0#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.tgt#0 <-> obj%p1#0: NoAlias
// CHECK-DAG: obj%p1#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.fir#0 <-> obj%p0.tgt.fir#0: NoAlias
// CHECK-DAG: obj%p0.fir#0 <-> obj%p1.tgt.fir#0: NoAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%p1.fir#0: NoAlias
// CHECK-DAG: obj%p1.fir#0 <-> obj%p1.tgt.fir#0: NoAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see the address of obj%arr(1) as an address
// that was loaded from a pointer and that could alias something.  However,
// t_obj%arr is a target.
// TODO: Thus, we expect the first case (and corresponding .fir case) below to
// be NoAlias.  However, the addresses obj%p0.tgt and obj%arr(1) are analyzed as
// MayAlias because they have the same source and both are data.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%arr(1)#0: MayAlias
// CHECK-DAG: obj%p0.tgt#0 <-> t_obj%arr(1)#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%arr(1).fir#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> t_obj%arr(1).fir#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_obj%alloc is a target.
// TODO: Thus, we expect the first case (and corresponding .fir case) below to
// be NoAlias.  However, the addresses obj%p0.tgt and obj%alloc.tgt are analyzed
// as MayAlias because they have the same source and both are data.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%p0.tgt#0 <-> t_obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%alloc.tgt.fir#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> t_obj%alloc.tgt.fir#0: MayAlias

// The address in an allocatable cannot alias the address of that allocatable.
// CHECK-DAG: obj%alloc#0 <-> obj%alloc.tgt#0: NoAlias
// CHECK-DAG: t_obj%alloc#0 <-> t_obj%alloc.tgt#0: NoAlias
// CHECK-DAG: obj%alloc.fir#0 <-> obj%alloc.tgt.fir#0: NoAlias
// CHECK-DAG: t_obj%alloc.fir#0 <-> t_obj%alloc.tgt.fir#0: NoAlias

// The address of a composite aliases the address of any component but not the
// address in a pointer or allocatable component.
// TODO: Thus, we expect the obj%*.tgt cases below to be NoAlias.  However, the
// addresses obj and obj%*.tgt are analyzed as MayAlias because they have the
// same source and both are data.
// CHECK-DAG: obj#0 <-> obj%p0#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%alloc#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%p0.tgt#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%p0.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%alloc.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%p0.tgt.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%alloc.tgt.fir#0: MayAlias

// The addresses obtained via multiple load instructions from the same
// allocatable can alias.
// CHECK-DAG: obj%alloc.tgt#0 <-> obj%alloc.tgt2#0: MayAlias
// CHECK-DAG: obj%alloc.tgt.fir#0 <-> obj%alloc.tgt2.fir#0: MayAlias

func.func @_QPtest() {
  %0 = fir.alloca !fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}> {bindc_name = "obj", uniq_name = "_QFtestEobj"}
  %1:2 = hlfir.declare %0 {test.ptr="obj", uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>)
  %2 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %3:2 = hlfir.declare %2 {test.ptr="t", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.alloca !fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}> {bindc_name = "t_obj", fir.target, uniq_name = "_QFtestEt_obj"}
  %5:2 = hlfir.declare %4 {test.ptr="t_obj", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_obj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>)
  %6 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %7:2 = hlfir.declare %6 {test.ptr="v", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %8 = hlfir.designate %1#0{"p0"}   {test.ptr="obj%p0", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %9 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.box_addr %9 {test.ptr="obj%p0.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %11 = hlfir.designate %1#0{"p1"}   {test.ptr="obj%p1", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %12 = fir.load %11 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %13 = fir.box_addr %12 {test.ptr="obj%p1.tgt"}: (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c2 = arith.constant 2 : index
  %14 = fir.shape %c2 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %15 = hlfir.designate %1#0{"arr"} <%14> (%c1) {test.ptr="obj%arr(1)"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %16 = hlfir.designate %1#0{"alloc"}   {test.ptr="obj%alloc", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %17 = fir.load %16 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %repeat17 = fir.load %16 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %18 = fir.box_addr %17 {test.ptr="obj%alloc.tgt"}: (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %repeat18 = fir.box_addr %repeat17 {test.ptr="obj%alloc.tgt2"}: (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %c2_1 = arith.constant 2 : index
  %19 = fir.shape %c2_1 : (index) -> !fir.shape<1>
  %c1_2 = arith.constant 1 : index
  %20 = hlfir.designate %5#0{"arr"} <%19> (%c1_2) {test.ptr="t_obj%arr(1)"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %21 = hlfir.designate %5#0{"alloc"}   {test.ptr="t_obj%alloc", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.load %21 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %23 = fir.box_addr %22 {test.ptr="t_obj%alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

func.func @_QPtest.fir() {
  %0 = fir.alloca !fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}> {bindc_name = "obj", uniq_name = "_QFtestEobj"}
  %1 = fir.declare %0 {test.ptr="obj.fir", uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %2 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %3 = fir.declare %2 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %4 = fir.alloca !fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}> {bindc_name = "t_obj", fir.target, uniq_name = "_QFtestEt_obj"}
  %5 = fir.declare %4 {test.ptr="t_obj.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_obj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %6 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %7 = fir.declare %6 {test.ptr = "v.fir", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.coordinate_of %1, p0 {test.ptr="obj%p0.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.box_addr %10 {test.ptr = "obj%p0.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %13 = fir.coordinate_of %1, p1 {test.ptr="obj%p1.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %14 = fir.load %13 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %15 = fir.box_addr %14 {test.ptr = "obj%p1.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c2 = arith.constant 2 : index
  %16 = fir.shape %c2 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %18 = fir.coordinate_of %1, arr : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.array<2xf32>>
  %19 = fir.array_coor %18(%16) %c1 {test.ptr="obj%arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %21 = fir.coordinate_of %1, alloc {test.ptr="obj%alloc.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.load %21 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %repeat22 = fir.load %21 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %23 = fir.box_addr %22 {test.ptr = "obj%alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %repeat23 = fir.box_addr %repeat22 {test.ptr = "obj%alloc.tgt2.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %c2_0 = arith.constant 2 : index
  %24 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %c1_1 = arith.constant 1 : index
  %26 = fir.coordinate_of %5, arr : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.array<2xf32>>
  %27 = fir.array_coor %26(%24) %c1_1 {test.ptr="t_obj%arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %29 = fir.coordinate_of %5, alloc {test.ptr="t_obj%alloc.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %30 = fir.load %29 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %31 = fir.box_addr %30 {test.ptr = "t_obj%alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

// -----

// Repeat above test except composites are dummy args instead of locals.

// module m
//   type :: ty
//     real, pointer :: p0, p1
//     real :: arr(2)
//     real, allocatable :: alloc
//     ! target attribute on components is not supported
//   end type ty
// end module m
// subroutine test(obj, t_obj)
//   use m
//   type(ty) :: obj
//   type(ty), target :: t_obj
//   real, target :: t
//   real :: v
// end subroutine test

// CHECK-LABEL: Testing : "_QPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%p1.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> obj%p0.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> obj%p1.tgt#0: MayAlias
// CHECK-DAG: v#0 <-> obj%p0.tgt#0: NoAlias
// CHECK-DAG: v#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%p1.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> obj%p0.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> obj%p1.tgt.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> obj%p0.tgt.fir#0: NoAlias
// CHECK-DAG: v.fir#0 <-> obj%p1.tgt.fir#0: NoAlias

// The address in a pointer cannot alias the address of a pointer.
// CHECK-DAG: obj%p0#0 <-> obj%p0.tgt#0: NoAlias
// CHECK-DAG: obj%p0#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.tgt#0 <-> obj%p1#0: NoAlias
// CHECK-DAG: obj%p1#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.fir#0 <-> obj%p0.tgt.fir#0: NoAlias
// CHECK-DAG: obj%p0.fir#0 <-> obj%p1.tgt.fir#0: NoAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%p1.fir#0: NoAlias
// CHECK-DAG: obj%p1.fir#0 <-> obj%p1.tgt.fir#0: NoAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see the address of obj%arr(1) as an address
// that was loaded from a pointer and that could alias something.  However,
// t_obj%arr is a target.
// TODO: Thus, we expect the first case (and corresponding .fir case) below to
// be NoAlias.  However, the addresses obj%p0.tgt and obj%arr(1) are analyzed as
// MayAlias because they have the same source and both are data.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%arr(1)#0: MayAlias
// CHECK-DAG: obj%p0.tgt#0 <-> t_obj%arr(1)#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%arr(1).fir#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> t_obj%arr(1).fir#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_obj%alloc is a target.
// TODO: Thus, we expect the first case (and corresponding .fir case) below to
// be NoAlias.  However, the addresses obj%p0.tgt and obj%alloc.tgt are analyzed
// as MayAlias because they have the same source and both are data.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%p0.tgt#0 <-> t_obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%alloc.tgt.fir#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> t_obj%alloc.tgt.fir#0: MayAlias

// The address in an allocatable cannot alias the address of that allocatable.
// TODO: Thus, we expect all cases below to be NoAlias.  However, target dummy
// args are currently indiscrimnately analyzed as MayAlias.
// CHECK-DAG: obj%alloc#0 <-> obj%alloc.tgt#0: NoAlias
// CHECK-DAG: t_obj%alloc#0 <-> t_obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%alloc.fir#0 <-> obj%alloc.tgt.fir#0: NoAlias
// CHECK-DAG: t_obj%alloc.fir#0 <-> t_obj%alloc.tgt.fir#0: MayAlias

// The address of a composite aliases the address of any component but not the
// address in a pointer or allocatable component.
// TODO: Thus, we expect the obj%*.tgt cases below to be NoAlias.  However, the
// addresses obj and obj%*.tgt are analyzed as MayAlias because they have the
// same source and both are data.
// CHECK-DAG: obj#0 <-> obj%p0#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%alloc#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%p0.tgt#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%p0.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%alloc.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%p0.tgt.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%alloc.tgt.fir#0: MayAlias

// The addresses obtained via multiple load instructions from the same
// allocatable can alias.
// CHECK-DAG: obj%alloc.tgt#0 <-> obj%alloc.tgt2#0: MayAlias
// CHECK-DAG: obj%alloc.tgt.fir#0 <-> obj%alloc.tgt2.fir#0: MayAlias

func.func @_QPtest(%arg0: !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>> {fir.bindc_name = "obj"}, %arg1: !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>> {fir.bindc_name = "t_obj", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="obj", uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>)
  %2 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %3:2 = hlfir.declare %2 {test.ptr="t", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4:2 = hlfir.declare %arg1 dummy_scope %0 {test.ptr="t_obj", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_obj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>)
  %5 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %6:2 = hlfir.declare %5 {test.ptr="v", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %7 = hlfir.designate %1#0{"p0"}   {test.ptr="obj%p0", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %8 = fir.load %7 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %9 = fir.box_addr %8 {test.ptr="obj%p0.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %10 = hlfir.designate %1#0{"p1"}   {test.ptr="obj%p1", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.load %10 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %12 = fir.box_addr %11 {test.ptr="obj%p1.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c2 = arith.constant 2 : index
  %13 = fir.shape %c2 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %14 = hlfir.designate %1#0{"arr"} <%13> (%c1) {test.ptr="obj%arr(1)"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %15 = hlfir.designate %1#0{"alloc"}   {test.ptr="obj%alloc", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %16 = fir.load %15 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %repeat16 = fir.load %15 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %17 = fir.box_addr %16 {test.ptr="obj%alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %repeat17 = fir.box_addr %repeat16 {test.ptr="obj%alloc.tgt2"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %c2_0 = arith.constant 2 : index
  %18 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %c1_1 = arith.constant 1 : index
  %19 = hlfir.designate %4#0{"arr"} <%18> (%c1_1) {test.ptr="t_obj%arr(1)"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %20 = hlfir.designate %4#0{"alloc"}   {test.ptr="t_obj%alloc", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %21 = fir.load %20 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.box_addr %21 {test.ptr="t_obj%alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

func.func @_QPtest.fir(%arg0: !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>> {fir.bindc_name = "obj"}, %arg1: !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>> {fir.bindc_name = "t_obj", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {test.ptr="obj.fir", uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %2 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %3 = fir.declare %2 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %4 = fir.declare %arg1 dummy_scope %0 {test.ptr="t_obj.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt_obj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %5 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %6 = fir.declare %5 {test.ptr = "v.fir", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %8 = fir.coordinate_of %1, p0 {test.ptr="obj%p0.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %9 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.box_addr %9 {test.ptr = "obj%p0.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %12 = fir.coordinate_of %1, p1 {test.ptr="obj%p1.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %13 = fir.load %12 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %14 = fir.box_addr %13 {test.ptr = "obj%p1.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c2 = arith.constant 2 : index
  %15 = fir.shape %c2 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %17 = fir.coordinate_of %1, arr : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.array<2xf32>>
  %18 = fir.array_coor %17(%15) %c1 {test.ptr="obj%arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %20 = fir.coordinate_of %1, alloc {test.ptr="obj%alloc.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %21 = fir.load %20 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %repeat21 = fir.load %20 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.box_addr %21 {test.ptr = "obj%alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %repeat22 = fir.box_addr %repeat21 {test.ptr = "obj%alloc.tgt2.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %c2_0 = arith.constant 2 : index
  %23 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %c1_1 = arith.constant 1 : index
  %25 = fir.coordinate_of %4, arr : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.array<2xf32>>
  %26 = fir.array_coor %25(%23) %c1_1 {test.ptr="t_obj%arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %28 = fir.coordinate_of %4, alloc {test.ptr="t_obj%alloc.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %29 = fir.load %28 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %30 = fir.box_addr %29 {test.ptr = "t_obj%alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

// -----

// Repeat above test except composites are globals.

// module m
//   type :: ty
//     real, pointer :: p0, p1
//     real :: arr(2)
//     real, allocatable :: alloc
//     ! target attribute on components is not supported
//   end type ty
//   type(ty) :: obj
//   type(ty), target :: t_obj
// end module m
// subroutine test()
//   use m
//   real, target :: t
//   real :: v
// end subroutine test

// CHECK-LABEL: Testing : "_QPtest"

// The address in a pointer can alias the address in another pointer or the
// address of a target but not the address of other variables.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%p1.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> obj%p0.tgt#0: MayAlias
// CHECK-DAG: t#0 <-> obj%p1.tgt#0: MayAlias
// CHECK-DAG: v#0 <-> obj%p0.tgt#0: NoAlias
// CHECK-DAG: v#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%p1.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> obj%p0.tgt.fir#0: MayAlias
// CHECK-DAG: t.fir#0 <-> obj%p1.tgt.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> obj%p0.tgt.fir#0: NoAlias
// CHECK-DAG: v.fir#0 <-> obj%p1.tgt.fir#0: NoAlias

// The address in a pointer cannot alias the address of a pointer.
// CHECK-DAG: obj%p0#0 <-> obj%p0.tgt#0: NoAlias
// CHECK-DAG: obj%p0#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.tgt#0 <-> obj%p1#0: NoAlias
// CHECK-DAG: obj%p1#0 <-> obj%p1.tgt#0: NoAlias
// CHECK-DAG: obj%p0.fir#0 <-> obj%p0.tgt.fir#0: NoAlias
// CHECK-DAG: obj%p0.fir#0 <-> obj%p1.tgt.fir#0: NoAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%p1.fir#0: NoAlias
// CHECK-DAG: obj%p1.fir#0 <-> obj%p1.tgt.fir#0: NoAlias

// For some cases, AliasAnalysis analyzes hlfir.designate like fir.box_addr, so
// make sure it doesn't mistakenly see the address of obj%arr(1) as an address
// that was loaded from a pointer and that could alias something.  However,
// t_obj%arr is a target.
// TODO: Thus, we expect the first case (and corresponding .fir case) below to
// be NoAlias.  However, the addresses obj%p0.tgt and obj%arr(1) are analyzed as
// MayAlias because they have the same source and both are data.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%arr(1)#0: MayAlias
// CHECK-DAG: obj%p0.tgt#0 <-> t_obj%arr(1)#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%arr(1).fir#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> t_obj%arr(1).fir#0: MayAlias

// Like a pointer, an allocatable contains an address, but an allocatable is not
// a pointer and so cannot alias pointers.  However, t_obj%alloc is a target.
// TODO: Thus, we expect the first case (and corresponding .fir case) below to
// be NoAlias.  However, the addresses obj%p0.tgt and obj%alloc.tgt are analyzed
// as MayAlias because they have the same source and both are data.
// CHECK-DAG: obj%p0.tgt#0 <-> obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%p0.tgt#0 <-> t_obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> obj%alloc.tgt.fir#0: MayAlias
// CHECK-DAG: obj%p0.tgt.fir#0 <-> t_obj%alloc.tgt.fir#0: MayAlias

// The address in an allocatable cannot alias the address of that allocatable.
// CHECK-DAG: obj%alloc#0 <-> obj%alloc.tgt#0: NoAlias
// CHECK-DAG: t_obj%alloc#0 <-> t_obj%alloc.tgt#0: NoAlias
// CHECK-DAG: obj%alloc.fir#0 <-> obj%alloc.tgt.fir#0: NoAlias
// CHECK-DAG: t_obj%alloc.fir#0 <-> t_obj%alloc.tgt.fir#0: NoAlias

// The address of a composite aliases the address of any component but not the
// address in a pointer or allocatable component.
// TODO: Thus, we expect the obj%*.tgt cases below to be NoAlias.  However, the
// addresses obj and obj%*.tgt are analyzed as MayAlias because they have the
// same source and both are data.
// CHECK-DAG: obj#0 <-> obj%p0#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%alloc#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%p0.tgt#0: MayAlias
// CHECK-DAG: obj#0 <-> obj%alloc.tgt#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%p0.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%alloc.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%p0.tgt.fir#0: MayAlias
// CHECK-DAG: obj.fir#0 <-> obj%alloc.tgt.fir#0: MayAlias

// The addresses obtained via multiple load instructions from the same
// allocatable can alias.
// CHECK-DAG: obj%alloc.tgt#0 <-> obj%alloc.tgt2#0: MayAlias
// CHECK-DAG: obj%alloc.tgt.fir#0 <-> obj%alloc.tgt2.fir#0: MayAlias

func.func @_QPtest() {
  %0 = fir.address_of(@_QMmEobj) : !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %1:2 = hlfir.declare %0 {test.ptr="obj", uniq_name = "_QMmEobj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>)
  %2 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %3:2 = hlfir.declare %2 {test.ptr="t", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.address_of(@_QMmEt_obj) : !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %5:2 = hlfir.declare %4 {test.ptr="t_obj", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEt_obj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>)
  %6 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %7:2 = hlfir.declare %6 {test.ptr="v", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %8 = hlfir.designate %1#0{"p0"}   {test.ptr="obj%p0", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %9 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.box_addr %9 {test.ptr="obj%p0.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %12 = hlfir.designate %1#0{"p1"}   {test.ptr="obj%p1", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %13 = fir.load %12 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %14 = fir.box_addr %13 {test.ptr="obj%p1.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c2 = arith.constant 2 : index
  %16 = fir.shape %c2 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %17 = hlfir.designate %1#0{"arr"} <%16> (%c1) {test.ptr="obj%arr(1)"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %19 = hlfir.designate %1#0{"alloc"}   {test.ptr="obj%alloc", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %20 = fir.load %19 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %repeat20 = fir.load %19 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %21 = fir.box_addr %20 {test.ptr="obj%alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %repeat21 = fir.box_addr %repeat20 {test.ptr="obj%alloc.tgt2"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %c2_0 = arith.constant 2 : index
  %23 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %c1_1 = arith.constant 1 : index
  %24 = hlfir.designate %5#0{"arr"} <%23> (%c1_1) {test.ptr="t_obj%arr(1)"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %26 = hlfir.designate %5#0{"alloc"}   {test.ptr="t_obj%alloc", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %27 = fir.load %26 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %28 = fir.box_addr %27 {test.ptr="t_obj%alloc.tgt"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}

func.func @_QPtest.fir() {
  %0 = fir.address_of(@_QMmEobj) : !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %1 = fir.declare %0 {test.ptr="obj.fir", uniq_name = "_QMmEobj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %2 = fir.alloca f32 {bindc_name = "t", fir.target, uniq_name = "_QFtestEt"}
  %3 = fir.declare %2 {test.ptr = "t.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %4 = fir.address_of(@_QMmEt_obj) : !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %5 = fir.declare %4 {test.ptr="t_obj.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEt_obj"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>
  %6 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtestEv"}
  %7 = fir.declare %6 {test.ptr = "v.fir", uniq_name = "_QFtestEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.coordinate_of %1, p0 {test.ptr="obj%p0.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.box_addr %10 {test.ptr = "obj%p0.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %13 = fir.coordinate_of %1, p1 {test.ptr="obj%p1.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %14 = fir.load %13 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %15 = fir.box_addr %14 {test.ptr = "obj%p1.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %c2 = arith.constant 2 : index
  %16 = fir.shape %c2 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %18 = fir.coordinate_of %1, arr : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.array<2xf32>>
  %19 = fir.array_coor %18(%16) %c1 {test.ptr="obj%arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %21 = fir.coordinate_of %1, alloc {test.ptr="obj%alloc.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.load %21 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %repeat22 = fir.load %21 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %23 = fir.box_addr %22 {test.ptr = "obj%alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %repeat23 = fir.box_addr %repeat22 {test.ptr = "obj%alloc.tgt2.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %c2_0 = arith.constant 2 : index
  %24 = fir.shape %c2_0 : (index) -> !fir.shape<1>
  %c1_1 = arith.constant 1 : index
  %26 = fir.coordinate_of %5, arr : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.array<2xf32>>
  %27 = fir.array_coor %26(%24) %c1_1 {test.ptr="t_obj%arr(1).fir"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %29 = fir.coordinate_of %5, alloc {test.ptr="t_obj%alloc.fir"} : (!fir.ref<!fir.type<_QMmTty{p0:!fir.box<!fir.ptr<f32>>,p1:!fir.box<!fir.ptr<f32>>,arr:!fir.array<2xf32>,alloc:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %30 = fir.load %29 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %31 = fir.box_addr %30 {test.ptr = "t_obj%alloc.tgt.fir"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  return
}
