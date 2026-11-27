// Check AliasAnalysis for pointer components.
//
// Throughout this test, The ".fir" suffix on symbols indicates a version of the
// MLIR after convert-hlfir-to-fir.  A key difference is that component access
// is via fir.coordinate_of instead of hlfir.designate.  We would like alias
// analysis results to be the same in both versions.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading  \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// module m
// type t
//  type(t), pointer :: next
//  integer :: i
// end type
// contains
// subroutine foo(x, y)
//   type(t) :: x, y
//   integer :: i1, i2
//   i1 = x%next%i
//   x = y
//   i2 = x%next%i
// end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPfoo"
// x and y are non pointer, non target argument and therefore do not alias.
// CHECK-DAG: x#0 <-> y#0: NoAlias
// CHECK-DAG: x.fir#0 <-> y.fir#0: NoAlias

// y is not a pointer object and therefore does not alias with the x%next
// component.  Also assigning x to y would not modify x.next
// CHECK-DAG: y#0 <-> xnext1#0: NoAlias
// CHECK-DAG: y#0 <-> xnext2#0: NoAlias
// CHECK-DAG: y.fir#0 <-> xnext1.fir#0: NoAlias
// CHECK-DAG: y.fir#0 <-> xnext2.fir#0: NoAlias

// We need to catch the fact that assigning y to x will modify xnext. 
// The only side-effect between the 2 loads of x.next is the assignment to x, 
// therefore x needs to alias with x.next to prevent the loads from being merged.
// CHECK-DAG: x#0 <-> xnext1#0: MayAlias
// CHECK-DAG: x#0 <-> xnext2#0: MayAlias
// CHECK-DAG: x.fir#0 <-> xnext1.fir#0: MayAlias
// CHECK-DAG: x.fir#0 <-> xnext2.fir#0: MayAlias

// TODO: xnext1#0 <-> xnext2#0 are the same and therefore MustAlias but 
//       we are currently not comparing operands involved in offset computations
// CHECK-DAG: xnext1#0 <-> xnext2#0: MayAlias
// CHECK-DAG: xnext1.fir#0 <-> xnext2.fir#0: MayAlias

// CHECK-DAG: xnext1#0 <-> ynext#0: NoAlias
// CHECK-DAG: xnext1.fir#0 <-> ynext.fir#0: NoAlias

func.func @_QMmPfoo(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %0 = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFfooEi1"}
  %1:2 = hlfir.declare %0 uniq_name("_QMmFfooEi1") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QMmFfooEi2"}
  %3:2 = hlfir.declare %2 uniq_name("_QMmFfooEi2") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg0 uniq_name("_QMmFfooEx") {test.ptr = "x"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %5:2 = hlfir.declare %arg1 uniq_name("_QMmFfooEy") {test.ptr = "y"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %6 = hlfir.designate %4#0{"next"}   fortran_attrs<pointer> {test.ptr = "xnext1"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %9 = hlfir.designate %8{"i"} : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  hlfir.assign %10 to %1#0 : i32, !fir.ref<i32>
  hlfir.assign %5#0 to %4#0 : !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %11 = hlfir.designate %4#0{"next"}   fortran_attrs<pointer> {test.ptr = "xnext2"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %12 = fir.load %11 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %13 = fir.box_addr %12 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %14 = hlfir.designate %13{"i"} : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %15 = fir.load %14 : !fir.ref<i32>
  hlfir.assign %15 to %3#0 : i32, !fir.ref<i32>
  %16 = hlfir.designate %5#0{"next"}   fortran_attrs<pointer> {test.ptr = "ynext"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  return
}

func.func @_QMmPfoo.fir(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %0 = fir.alloca !fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %1 = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFfooEi1"}
  %2 = fir.declare %1 uniq_name("_QMmFfooEi1") : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QMmFfooEi2"}
  %4 = fir.declare %3 uniq_name("_QMmFfooEi2") : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg0 uniq_name("_QMmFfooEx") {test.ptr = "x.fir"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %6 = fir.declare %arg1 uniq_name("_QMmFfooEy") {test.ptr = "y.fir"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %8 = fir.coordinate_of %5, next {test.ptr="xnext1.fir"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %9 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %10 = fir.box_addr %9 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %12 = fir.coordinate_of %10, i : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %13 = fir.load %12 : !fir.ref<i32>
  fir.store %13 to %2 : !fir.ref<i32>
  %14 = fir.embox %5 : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %15 = fir.embox %6 : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  fir.store %14 to %0 : !fir.ref<!fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>
  %16 = fir.address_of(@_QQclX746D70332E66697200) : !fir.ref<!fir.char<1,9>>
  %c9 = arith.constant 9 : index
  %c14_i32 = arith.constant 14 : i32
  %17 = fir.convert %0 : (!fir.ref<!fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ref<!fir.box<none>>
  %18 = fir.convert %15 : (!fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.box<none>
  %19 = fir.convert %16 : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<i8>
  fir.call @_FortranAAssign(%17, %18, %19, %c14_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  %22 = fir.coordinate_of %5, next {test.ptr="xnext2.fir"}: (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %23 = fir.load %22 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %24 = fir.box_addr %23 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %26 = fir.coordinate_of %24, i : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %27 = fir.load %26 : !fir.ref<i32>
  fir.store %27 to %4 : !fir.ref<i32>
  %29 = fir.coordinate_of %6, next {test.ptr="ynext.fir"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  return
}

// -----

// Same test as above focusing on aliasing between x%next and y%next data
// CHECK-LABEL: Testing : "_QMmPfoo2"
// CHECK-DAG: xnext#0 <-> ynext#0: MayAlias
// CHECK-DAG: xnext.fir#0 <-> ynext.fir#0: MayAlias

func.func @_QMmPfoo2(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %4:2 = hlfir.declare %arg0 uniq_name("_QMmFfooEx") : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %5:2 = hlfir.declare %arg1 uniq_name("_QMmFfooEy") : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %6 = hlfir.designate %4#0{"next"}   fortran_attrs<pointer> : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %8 = fir.box_addr %7 {test.ptr = "xnext"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>

  %9 = hlfir.designate %5#0{"next"}   fortran_attrs<pointer> : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %11 = fir.box_addr %10 {test.ptr = "ynext"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  return
}

func.func @_QMmPfoo2.fir(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %0 = fir.declare %arg0 uniq_name("_QMmFfooEx") : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %1 = fir.declare %arg1 uniq_name("_QMmFfooEy") : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %3 = fir.coordinate_of %0, next : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %5 = fir.box_addr %4 {test.ptr = "xnext.fir"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %7 = fir.coordinate_of %1, next : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %8 = fir.load %7 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %9 = fir.box_addr %8 {test.ptr = "ynext.fir"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  return
}

// -----

// module m
// type ta
//  integer, pointer :: array(:)
// end type
// contains
// subroutine foo3(x, y)
//   type(t) :: x, y
//   x%array(1) = y%array(1)
// end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPfoo3"
// CHECK-DAG: yarray#0 <-> xarray#0: MayAlias
// CHECK-DAG: yarray.fir#0 <-> xarray.fir#0: MayAlias

func.func @_QMmPfoo3(%arg0: !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>> {fir.bindc_name = "y"}) {
  %0:2 = hlfir.declare %arg0 uniq_name("_QMmFfooEx") : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>)
  %1:2 = hlfir.declare %arg1 uniq_name("_QMmFfooEy") : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>)
  %2 = hlfir.designate %1#0{"array"}   fortran_attrs<pointer> : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %3 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %4 = hlfir.designate %3 (%c1) {test.ptr = "yarray"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  %5 = fir.load %4 : !fir.ref<i32>
  %6 = hlfir.designate %0#0{"array"}   fortran_attrs<pointer> : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1_0 = arith.constant 1 : index
  %8 = hlfir.designate %7 (%c1_0)  {test.ptr = "xarray"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  hlfir.assign %5 to %8 : i32, !fir.ref<i32>
  return
}

func.func @_QMmPfoo3.fir(%arg0: !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>> {fir.bindc_name = "y"}) {
  %0 = fir.declare %arg0 uniq_name("_QMmFfooEx") : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
  %1 = fir.declare %arg1 uniq_name("_QMmFfooEy") : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
  %3 = fir.coordinate_of %1, array : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
  %6 = fir.shift %5#0 : (index) -> !fir.shift<1>
  %7 = fir.array_coor %4(%6) %c1 {test.ptr="yarray.fir"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
  %8 = fir.load %7 : !fir.ref<i32>
  %10 = fir.coordinate_of %0, array : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %11 = fir.load %10 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1_0 = arith.constant 1 : index
  %c0_1 = arith.constant 0 : index
  %12:3 = fir.box_dims %11, %c0_1 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
  %13 = fir.shift %12#0 : (index) -> !fir.shift<1>
  %14 = fir.array_coor %11(%13) %c1_0 {test.ptr="xarray.fir"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
  fir.store %8 to %14 : !fir.ref<i32>
  return
}

// -----

// The address of a composite aliases the address of any component, including an
// allocatable component.  Like the address of a pointer, the address of an
// allocatable is considered non-data, so AliasAnalysis has special handling to
// detect the aliasing.

// module m
//   type t
//     integer, allocatable :: p
//   end type
//   type(t) :: x
// contains
//   subroutine test()
//     ! access x%p
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest"
// CHECK-DAG: x#0 <-> x%p#0: MayAlias
// CHECK-DAG: x.fir#0 <-> x%p.fir#0: MayAlias

func.func @_QMmPtest() {
  %0 = fir.address_of(@_QMmEx) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>
  %1:2 = hlfir.declare %0 uniq_name("_QMmEx") {test.ptr = "x"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>)
  %2 = hlfir.designate %1#0{"p"}   fortran_attrs<allocatable> {test.ptr = "x%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  return
}

func.func @_QMmPtest.fir() {
  %0 = fir.address_of(@_QMmEx) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>
  %1 = fir.declare %0 uniq_name("_QMmEx") {test.ptr = "x.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>
  %3 = fir.coordinate_of %1, p {test.ptr="x%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  return
}

// -----

// Nested composites.

// module m
//   type t1
//     integer, pointer :: p
//   end type
//   type t2
//     type(t1) :: x
//     integer, pointer :: p
//     integer :: i
//   end type
// contains
//   subroutine test()
//     type(t2) :: x
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest"

// The addresses of a composite and its pointer component alias even if the
// composite is nested within another composite.
// CHECK-DAG: x#0 <-> x%p#0: MayAlias
// CHECK-DAG: x%x#0 <-> x%x%p#0: MayAlias
// CHECK-DAG: x.fir#0 <-> x%p.fir#0: MayAlias
// CHECK-DAG: x%x.fir#0 <-> x%x%p.fir#0: MayAlias

// The addresses of different components of the same composite do not alias.
// CHECK-DAG: x%x#0 <-> x%i#0: NoAlias
// CHECK-DAG: x%p#0 <-> x%i#0: NoAlias
// CHECK-DAG: x%x#0 <-> x%p#0: NoAlias
// CHECK-DAG: x%x.fir#0 <-> x%i.fir#0: NoAlias
// CHECK-DAG: x%p.fir#0 <-> x%i.fir#0: NoAlias
// CHECK-DAG: x%x.fir#0 <-> x%p.fir#0: NoAlias

func.func @_QMmPtest() {
  %0 = fir.alloca !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "x", uniq_name = "_QMmFtestEx"}
  %1:2 = hlfir.declare %0 uniq_name("_QMmFtestEx") {test.ptr = "x"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %2 = hlfir.designate %1#0{"x"}  {test.ptr="x%x"}  : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>
  %3 = hlfir.designate %1#0{"p"}   fortran_attrs<pointer> {test.ptr = "x%p"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %4 = hlfir.designate %1#0{"i"} {test.ptr="x%i"}  : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>
  %5 = hlfir.designate %2{"p"}   fortran_attrs<pointer> {test.ptr = "x%x%p"} : (!fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  return
}

func.func @_QMmPtest.fir() {
  %0 = fir.alloca !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "x", uniq_name = "_QMmFtestEx"}
  %1 = fir.declare %0 uniq_name("_QMmFtestEx") {test.ptr = "x.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %3 = fir.coordinate_of %1, x {test.ptr="x%x.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>
  %5 = fir.coordinate_of %1, p {test.ptr="x%p.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %7 = fir.coordinate_of %1, i {test.ptr="x%i.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>
  %9 = fir.coordinate_of %3, p {test.ptr="x%x%p.fir"} : (!fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  return
}

// -----

// Pointer that might dynamically be a component.

// The address of a pointer dummy arg (argp) might alias the address of a
// pointer component (x%p) and thus the address of the associated composite (x),
// but it does not alias the addresses of other components (x%i) of the
// composite.  Moreover, the address *in* argp does not alias any of those.
// Finally, an allocatable dummy arg (arga) should not be mistaken for a pointer
// dummy arg and cannot have such aliasing.
//
// Such an x can be a global (glob), but not a local (loc) or a dummy arg with
// neither TARGET nor POINTER (arg): only arg may change arg%p, and if it does,
// only arg may reference arg%p (F2023 15.5.2.14).

// module m
//   type t
//     integer, pointer :: p
//     integer i
//   end type
//   type(t) :: glob
// contains
//   subroutine test(argp, arga, arg)
//     integer, pointer :: argp
//     integer, allocatable :: arga
//     type(t) :: arg
//     type(t) :: loc
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest"

// Check when composite is a dummy arg.
//
// CHECK-DAG: argp#0 <-> arg#0: NoAlias
// CHECK-DAG: argp#0 <-> arg%p#0: NoAlias
// CHECK-DAG: argp#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg%p.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg%i.fir#0: NoAlias
//
// CHECK-DAG: argp.tgt#0 <-> arg#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg%p.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg%i.fir#0: NoAlias
//
// CHECK-DAG: arga#0 <-> arg#0: NoAlias
// CHECK-DAG: arga#0 <-> arg%p#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> arg.fir#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> arg%p.fir#0: NoAlias

// Check when composite is a global.
//
// CHECK-DAG: argp#0 <-> glob#0: MayAlias
// CHECK-DAG: argp#0 <-> glob%p#0: MayAlias
// CHECK-DAG: argp#0 <-> glob%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> glob.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> glob%p.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> glob%i.fir#0: NoAlias
//
// CHECK-DAG: argp.tgt#0 <-> glob#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob%i#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob%p.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob%i.fir#0: NoAlias
//
// CHECK-DAG: arga#0 <-> glob#0: NoAlias
// CHECK-DAG: arga#0 <-> glob%p#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> glob.fir#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> glob%p.fir#0: NoAlias

// Check when composite is a local and thus cannot alias a dummy arg.
//
// CHECK-DAG: argp#0 <-> loc#0: NoAlias
// CHECK-DAG: argp#0 <-> loc%p#0: NoAlias
// CHECK-DAG: argp#0 <-> loc%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> loc.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> loc%p.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> loc%i.fir#0: NoAlias
//
// CHECK-DAG: argp.tgt#0 <-> loc#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc%i#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc%p.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc%i.fir#0: NoAlias
//
// CHECK-DAG: arga#0 <-> loc#0: NoAlias
// CHECK-DAG: arga#0 <-> loc%p#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> loc.fir#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> loc%p.fir#0: NoAlias

fir.global @_QMmEglob : !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>

func.func @_QMmPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "arga"}, %arg2: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}) {
  %0 = fir.dummy_scope : !fir.dscope

  %1:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QMmFtestEargp") fortran_attrs<pointer> {test.ptr = "argp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.box_addr %2 {test.ptr="argp.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %4:2 = hlfir.declare %arg1 dummy_scope %0 uniq_name("_QMmFtestEarga") fortran_attrs<allocatable> {test.ptr = "arga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)

  %5:2 = hlfir.declare %arg2 dummy_scope %0 uniq_name("_QMmFtestEarg") {test.ptr = "arg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %6 = hlfir.designate %5#0{"p"}   fortran_attrs<pointer> {test.ptr = "arg%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %7 = hlfir.designate %5#0{"i"} {test.ptr="arg%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %8 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %9:2 = hlfir.declare %8 uniq_name("_QMmEglob") {test.ptr = "glob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %10 = hlfir.designate %9#0{"p"}   fortran_attrs<pointer> {test.ptr = "glob%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %11 = hlfir.designate %9#0{"i"} {test.ptr="glob%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %12 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFtestEloc"}
  %13:2 = hlfir.declare %12 uniq_name("_QMmFtestEloc") {test.ptr = "loc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %14 = hlfir.designate %13#0{"p"}   fortran_attrs<pointer> {test.ptr = "loc%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %15 = hlfir.designate %13#0{"i"} {test.ptr="loc%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}

func.func @_QMmPtest.fir(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "arga"}, %arg2: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}) {
  %0 = fir.dummy_scope : !fir.dscope

  %1 = fir.declare %arg0 dummy_scope %0 uniq_name("_QMmFtestEargp") fortran_attrs<pointer> {test.ptr = "argp.fir"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.box_addr %2 {test.ptr = "argp.tgt.fir"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %4 = fir.declare %arg1 dummy_scope %0 uniq_name("_QMmFtestEarga") fortran_attrs<allocatable> {test.ptr = "arga.fir"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %5 = fir.declare %arg2 dummy_scope %0 uniq_name("_QMmFtestEarg") {test.ptr = "arg.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %7 = fir.coordinate_of %5, p {test.ptr="arg%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %9 = fir.coordinate_of %5, i {test.ptr="arg%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %10 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %11 = fir.declare %10 uniq_name("_QMmEglob") {test.ptr = "glob.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %13 = fir.coordinate_of %11, p {test.ptr="glob%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %15 = fir.coordinate_of %11, i {test.ptr="glob%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %16 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFtestEloc"}
  %17 = fir.declare %16 uniq_name("_QMmFtestEloc") {test.ptr = "loc.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %19 = fir.coordinate_of %17, p {test.ptr="loc%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %21 = fir.coordinate_of %17, i {test.ptr="loc%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}

// -----

// Same as previous case but argp and arga are HostAssoc.

// module m
//   type t
//     integer, pointer :: p
//     integer i
//   end type
//   type(t) :: glob
// contains
//   subroutine parent(argp, arga)
//     integer, pointer :: argp
//     integer, allocatable :: arga
//     type(t) :: arg
//     call test(arg)
//   contains
//     subroutine test(arg)
//       type(t) :: arg
//       type(t) :: loc
//     end subroutine
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmFparentPtest"

// Check when composite is a dummy arg.
//
// CHECK-DAG: argp#0 <-> arg#0: NoAlias
// CHECK-DAG: argp#0 <-> arg%p#0: NoAlias
// CHECK-DAG: argp#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg%p.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg%i.fir#0: NoAlias
//
// Scope-aware refinement (ScopedOrigin) now disambiguates argp.tgt
// against unrelated scope-mate variables: arg is a non-TARGET dummy
// so a pointer component dereference cannot reach it.
// CHECK-DAG: argp.tgt#0 <-> arg#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg%p.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg%i.fir#0: NoAlias
//
// CHECK-DAG: arga#0 <-> arg#0: NoAlias
// CHECK-DAG: arga#0 <-> arg%p#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> arg.fir#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> arg%p.fir#0: NoAlias

// Check when composite is a global.
//
// CHECK-DAG: argp#0 <-> glob#0: MayAlias
// CHECK-DAG: argp#0 <-> glob%p#0: MayAlias
// CHECK-DAG: argp#0 <-> glob%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> glob.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> glob%p.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> glob%i.fir#0: NoAlias
//
// Scope-aware refinement (ScopedOrigin) now disambiguates argp.tgt
// against the non-TARGET global glob.
// CHECK-DAG: argp.tgt#0 <-> glob#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob%i#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob%p.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob%i.fir#0: NoAlias
//
// CHECK-DAG: arga#0 <-> glob#0: NoAlias
// CHECK-DAG: arga#0 <-> glob%p#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> glob.fir#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> glob%p.fir#0: NoAlias

// Check when composite is a local and thus cannot alias a dummy arg.
//
// CHECK-DAG: argp#0 <-> loc#0: NoAlias
// CHECK-DAG: argp#0 <-> loc%p#0: NoAlias
// CHECK-DAG: argp#0 <-> loc%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> loc.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> loc%p.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> loc%i.fir#0: NoAlias
//
// Scope-aware refinement (ScopedOrigin) now disambiguates argp.tgt
// against the non-TARGET local loc.
// CHECK-DAG: argp.tgt#0 <-> loc#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc%i#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc%p.fir#0: NoAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc%i.fir#0: NoAlias
//
// CHECK-DAG: arga#0 <-> loc#0: NoAlias
// CHECK-DAG: arga#0 <-> loc%p#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> loc.fir#0: NoAlias
// CHECK-DAG: arga.fir#0 <-> loc%p.fir#0: NoAlias

fir.global @_QMmEglob : !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>

func.func private @_QMmFparentPtest(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}, %arg1: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QMmPparent, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope

  %c0_i32 = arith.constant 0 : i32
  %1 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>
  %2 = fir.load %1 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>
  %3:2 = hlfir.declare %2 uniq_name("_QMmFparentEargp") fortran_attrs<pointer, host_assoc> {test.ptr = "argp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %4 = fir.load %3#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %5 = fir.box_addr %4 {test.ptr="argp.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %c1_i32 = arith.constant 1 : i32
  %10 = fir.coordinate_of %arg1, %c1_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %11 = fir.load %10 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %12:2 = hlfir.declare %11 uniq_name("_QMmFparentEarga") fortran_attrs<allocatable, host_assoc> {test.ptr = "arga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)

  %20 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %21:2 = hlfir.declare %20 uniq_name("_QMmEglob") {test.ptr = "glob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %22 = hlfir.designate %21#0{"p"}   fortran_attrs<pointer> {test.ptr = "glob%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %23 = hlfir.designate %21#0{"i"} {test.ptr="glob%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %30:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QMmFparentFtestEarg") {test.ptr = "arg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %31 = hlfir.designate %30#0{"p"}   fortran_attrs<pointer> {test.ptr = "arg%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %32 = hlfir.designate %30#0{"i"} {test.ptr="arg%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %40 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFparentFtestEloc"}
  %41:2 = hlfir.declare %40 uniq_name("_QMmFparentFtestEloc") {test.ptr = "loc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %42 = hlfir.designate %41#0{"p"}   fortran_attrs<pointer> {test.ptr = "loc%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %43 = hlfir.designate %41#0{"i"} {test.ptr="loc%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}

func.func private @_QMmFparentPtest.fir(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}, %arg1: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QMmPparent, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope

  %c0_i32 = arith.constant 0 : i32
  %1 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>
  %2 = fir.load %1 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>
  %3 = fir.declare %2 uniq_name("_QMmFparentEargp") fortran_attrs<pointer, host_assoc> {test.ptr = "argp.fir"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %5 = fir.box_addr %4 {test.ptr = "argp.tgt.fir"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %c1_i32 = arith.constant 1 : i32
  %6 = fir.coordinate_of %arg1, %c1_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %7 = fir.load %6 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %8 = fir.declare %7 uniq_name("_QMmFparentEarga") fortran_attrs<allocatable, host_assoc> {test.ptr = "arga.fir"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %9 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %10 = fir.declare %9 uniq_name("_QMmEglob") {test.ptr = "glob.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %12 = fir.coordinate_of %10, p {test.ptr="glob%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %14 = fir.coordinate_of %10, i {test.ptr="glob%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %15 = fir.declare %arg0 dummy_scope %0 uniq_name("_QMmFparentFtestEarg") {test.ptr = "arg.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %17 = fir.coordinate_of %15, p {test.ptr="arg%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %19 = fir.coordinate_of %15, i {test.ptr="arg%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %20 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFparentFtestEloc"}
  %21 = fir.declare %20 uniq_name("_QMmFparentFtestEloc") {test.ptr = "loc.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %23 = fir.coordinate_of %21, p {test.ptr="loc%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %25 = fir.coordinate_of %21, i {test.ptr="loc%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}

// -----

// Dummy arg with target attribute and pointer component.

// The address of a dummy arg (arg0) may alias the address of a global (glob0)
// or another dummy arg (arg1) when both have target attributes.  If either is a
// composite, the addresses of components (whether data like r, or non-data like
// p and a) may also alias the composite or the same component.  However, target
// attributes do not permit two globals (glob0 and glob1) to alias.

// module m
//   type t
//     real, pointer :: p
//     real, allocatable :: a
//     real :: r
//   end type
//   type(t), target :: glob0
//   type(t), target :: glob1
// contains
//   subroutine test(arg0, arg1)
//     type(t), target :: arg0
//     type(t), target :: arg1
//   end subroutine
// end module

// Check composite vs. composite.
//
// CHECK-DAG: arg0#0 <-> arg1#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> arg1.fir#0: MayAlias
//
// CHECK-DAG: arg0#0 <-> glob0#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> glob0.fir#0: MayAlias
//
// CHECK-DAG: glob0#0 <-> glob1#0: NoAlias
// CHECK-DAG: glob0.fir#0 <-> glob1.fir#0: NoAlias

// Check component vs. composite.
//
// CHECK-DAG: arg0%p#0 <-> arg1#0: MayAlias
// CHECK-DAG: arg0%a#0 <-> arg1#0: MayAlias
// CHECK-DAG: arg0%r#0 <-> arg1#0: MayAlias
// CHECK-DAG: arg0%p.fir#0 <-> arg1.fir#0: MayAlias
// CHECK-DAG: arg0%a.fir#0 <-> arg1.fir#0: MayAlias
// CHECK-DAG: arg0%r.fir#0 <-> arg1.fir#0: MayAlias
//
// CHECK-DAG: arg0%p#0 <-> glob0#0: MayAlias
// CHECK-DAG: arg0%a#0 <-> glob0#0: MayAlias
// CHECK-DAG: arg0%r#0 <-> glob0#0: MayAlias
// CHECK-DAG: arg0%p.fir#0 <-> glob0.fir#0: MayAlias
// CHECK-DAG: arg0%a.fir#0 <-> glob0.fir#0: MayAlias
// CHECK-DAG: arg0%r.fir#0 <-> glob0.fir#0: MayAlias
//
// CHECK-DAG: glob0%p#0 <-> glob1#0: NoAlias
// CHECK-DAG: glob0%a#0 <-> glob1#0: NoAlias
// CHECK-DAG: glob0%r#0 <-> glob1#0: NoAlias
// CHECK-DAG: glob0%p.fir#0 <-> glob1.fir#0: NoAlias
// CHECK-DAG: glob0%a.fir#0 <-> glob1.fir#0: NoAlias
// CHECK-DAG: glob0%r.fir#0 <-> glob1.fir#0: NoAlias

// Check composite vs. component.
//
// CHECK-DAG: arg0#0 <-> arg1%p#0: MayAlias
// CHECK-DAG: arg0#0 <-> arg1%a#0: MayAlias
// CHECK-DAG: arg0#0 <-> arg1%r#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> arg1%p.fir#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> arg1%a.fir#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> arg1%r.fir#0: MayAlias
//
// CHECK-DAG: arg0#0 <-> glob0%p#0: MayAlias
// CHECK-DAG: arg0#0 <-> glob0%a#0: MayAlias
// CHECK-DAG: arg0#0 <-> glob0%r#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> glob0%p.fir#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> glob0%a.fir#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> glob0%r.fir#0: MayAlias
//
// CHECK-DAG: glob0#0 <-> glob1%p#0: NoAlias
// CHECK-DAG: glob0#0 <-> glob1%a#0: NoAlias
// CHECK-DAG: glob0#0 <-> glob1%r#0: NoAlias
// CHECK-DAG: glob0.fir#0 <-> glob1%p.fir#0: NoAlias
// CHECK-DAG: glob0.fir#0 <-> glob1%a.fir#0: NoAlias
// CHECK-DAG: glob0.fir#0 <-> glob1%r.fir#0: NoAlias

// Check component vs. component.
//
// CHECK-DAG: arg0%p#0 <-> arg1%p#0: MayAlias
// CHECK-DAG: arg0%a#0 <-> arg1%a#0: MayAlias
// CHECK-DAG: arg0%r#0 <-> arg1%r#0: MayAlias
// CHECK-DAG: arg0%p.fir#0 <-> arg1%p.fir#0: MayAlias
// CHECK-DAG: arg0%a.fir#0 <-> arg1%a.fir#0: MayAlias
// CHECK-DAG: arg0%r.fir#0 <-> arg1%r.fir#0: MayAlias
//
// CHECK-DAG: arg0%p.fir#0 <-> glob0%p.fir#0: MayAlias
// CHECK-DAG: arg0%a.fir#0 <-> glob0%a.fir#0: MayAlias
// CHECK-DAG: arg0%r.fir#0 <-> glob0%r.fir#0: MayAlias
//
// CHECK-DAG: glob0%p.fir#0 <-> glob1%p.fir#0: NoAlias
// CHECK-DAG: glob0%a.fir#0 <-> glob1%a.fir#0: NoAlias
// CHECK-DAG: glob0%r.fir#0 <-> glob1%r.fir#0: NoAlias

func.func @_QMmPtest(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg0", fir.target}, %arg1: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg1", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope

  %10:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QMmFtestEarg0") fortran_attrs<target> {test.ptr = "arg0"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %11 = hlfir.designate %10#0{"p"}   fortran_attrs<pointer> {test.ptr = "arg0%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %12 = hlfir.designate %10#0{"a"}   fortran_attrs<allocatable> {test.ptr = "arg0%a"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %13 = hlfir.designate %10#0{"r"}   {test.ptr="arg0%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %20:2 = hlfir.declare %arg1 dummy_scope %0 uniq_name("_QMmFtestEarg1") fortran_attrs<target> {test.ptr = "arg1"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %21 = hlfir.designate %20#0{"p"}   fortran_attrs<pointer> {test.ptr = "arg1%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %22 = hlfir.designate %20#0{"a"}   fortran_attrs<allocatable> {test.ptr = "arg1%a"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %23 = hlfir.designate %20#0{"r"}   {test.ptr="arg1%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %30 = fir.address_of(@_QMmEglob0) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %31:2 = hlfir.declare %30 uniq_name("_QMmEglob0") fortran_attrs<target> {test.ptr = "glob0"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %32 = hlfir.designate %31#0{"p"}   fortran_attrs<pointer> {test.ptr = "glob0%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %33 = hlfir.designate %31#0{"a"}   fortran_attrs<allocatable> {test.ptr = "glob0%a"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %34 = hlfir.designate %31#0{"r"}   {test.ptr="glob0%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %40 = fir.address_of(@_QMmEglob1) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %41:2 = hlfir.declare %40 uniq_name("_QMmEglob1") fortran_attrs<target> {test.ptr = "glob1"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %42 = hlfir.designate %41#0{"p"}   fortran_attrs<pointer> {test.ptr = "glob1%p"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %43 = hlfir.designate %41#0{"a"}   fortran_attrs<allocatable> {test.ptr = "glob1%a"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %44 = hlfir.designate %41#0{"r"}   {test.ptr="glob1%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  return
}

func.func @_QMmPtest.fir(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg0", fir.target}, %arg1: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg1", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope

  %1 = fir.declare %arg0 dummy_scope %0 uniq_name("_QMmFtestEarg0") fortran_attrs<target> {test.ptr = "arg0.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %3 = fir.coordinate_of %1, p {test.ptr="arg0%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.coordinate_of %1, a {test.ptr="arg0%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %7 = fir.coordinate_of %1, r {test.ptr="arg0%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %8 = fir.declare %arg1 dummy_scope %0 uniq_name("_QMmFtestEarg1") fortran_attrs<target> {test.ptr = "arg1.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %10 = fir.coordinate_of %8, p {test.ptr="arg1%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %12 = fir.coordinate_of %8, a {test.ptr="arg1%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %14 = fir.coordinate_of %8, r {test.ptr="arg1%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %15 = fir.address_of(@_QMmEglob0) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %16 = fir.declare %15 uniq_name("_QMmEglob0") fortran_attrs<target> {test.ptr = "glob0.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %18 = fir.coordinate_of %16, p {test.ptr="glob0%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %20 = fir.coordinate_of %16, a {test.ptr="glob0%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.coordinate_of %16, r {test.ptr="glob0%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %23 = fir.address_of(@_QMmEglob1) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %24 = fir.declare %23 uniq_name("_QMmEglob1") fortran_attrs<target> {test.ptr = "glob1.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %26 = fir.coordinate_of %24, p {test.ptr="glob1%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %28 = fir.coordinate_of %24, a {test.ptr="glob1%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %30 = fir.coordinate_of %24, r {test.ptr="glob1%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  return
}

// -----

// Allocatable dummy arg with target attribute.

// This case is like the previous one except that the non-data is the address
// of the dummy arg itself rather than of a pointer component of the dummy arg.
// The goal is to check that logic introduced into AliasAnalysis to handle the
// pointer component case doesn't break this related one.

// module m
//   real, allocatable, target :: glob0
//   real, allocatable, target :: glob1
// contains
//   subroutine test(arg0, arg1)
//     real, allocatable, target :: arg0
//     real, allocatable, target :: arg1
//   end subroutine
// end module

// CHECK-DAG: arg0#0 <-> arg1#0: MayAlias
// CHECK-DAG: arg0#0 <-> glob0#0: MayAlias
// CHECK-DAG: glob0#0 <-> glob1#0: NoAlias

func.func @_QMmPtest(%arg0: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "arg0", fir.target}, %arg1: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "arg1", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope

  %10:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QMmFtestEarg0") fortran_attrs<allocatable, target> {test.ptr = "arg0"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %11:2 = hlfir.declare %arg1 dummy_scope %0 uniq_name("_QMmFtestEarg1") fortran_attrs<allocatable, target> {test.ptr = "arg1"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  %20 = fir.address_of(@_QMmEglob0) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %21:2 = hlfir.declare %20 uniq_name("_QMmEglob0") fortran_attrs<allocatable, target> {test.ptr = "glob0"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %22 = fir.address_of(@_QMmEglob1) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %23:2 = hlfir.declare %22 uniq_name("_QMmEglob1") fortran_attrs<allocatable, target> {test.ptr = "glob1"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  return
}

// -----

// Arrays of composites and descriptor values of composites.

// The address of an array of composites (loc) and a descriptor value of a
// composite or of an array of composites (arr, pol) alias the address of an
// allocatable component of an element, like the address of a scalar composite
// does.  A polymorphic object may have such a component even if its declared
// type has none (b).
//
// The data of an allocatable or pointer does not alias its descriptor if the
// data has no descriptor component (v), if the data is allocatable (w, s), or
// if the pointer is a whole local variable (q).  The target of a pointer may
// contain the pointer if the pointer is a component (xs(1)%p) or a dummy arg,
// whose actual arg may be a component (pd).

// module m
//   type t
//     integer, allocatable :: a
//   end type
//   type base
//     integer :: i
//   end type
//   type, extends(base) :: ext
//     integer, allocatable :: a
//   end type
//   type pt
//     integer :: x
//   end type
//   type tp
//     type(tp), pointer :: p(:)
//   end type
// contains
//   subroutine test_arrays(arr, pol, b, pd)
//     type(t) :: arr(:)
//     class(t) :: pol
//     class(base) :: b
//     type(tp), pointer :: pd(:)
//     type(t) :: loc(2)
//     type(t), allocatable :: w(:)
//     type(pt), allocatable :: v(:)
//     type(tp), pointer :: q(:)
//     type(tp), target :: xs(2)
//     type(t), allocatable :: s
//     ! access arr(1)%a, pol%a, loc(1)%a, w, v, q, pd, xs(1)%p and s
//     select type (b)
//     type is (ext)
//       ! access b%a
//     end select
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest_arrays"
// CHECK-DAG: arr#0 <-> arr(1)%a#0: MayAlias
// CHECK-DAG: pol#0 <-> pol%a#0: MayAlias
// CHECK-DAG: loc#0 <-> loc(1)%a#0: MayAlias
// CHECK-DAG: b#0 <-> b%a#0: MayAlias
// CHECK-DAG: v#0 <-> v.box#0: NoAlias
// CHECK-DAG: w#0 <-> w.box#0: NoAlias
// CHECK-DAG: q#0 <-> q.box#0: NoAlias
// CHECK-DAG: pd#0 <-> pd.box#0: MayAlias
// CHECK-DAG: xs(1)%p#0 <-> xs(1)%p.box#0: MayAlias
// CHECK-DAG: s#0 <-> s.addr#0: NoAlias
// CHECK-DAG: arr.fir#0 <-> arr(1)%a.fir#0: MayAlias
// CHECK-DAG: pol.fir#0 <-> pol%a.fir#0: MayAlias
// CHECK-DAG: loc.fir#0 <-> loc(1)%a.fir#0: MayAlias
// CHECK-DAG: b.fir#0 <-> b%a.fir#0: MayAlias
// CHECK-DAG: v.fir#0 <-> v.box.fir#0: NoAlias
// CHECK-DAG: w.fir#0 <-> w.box.fir#0: NoAlias
// CHECK-DAG: q.fir#0 <-> q.box.fir#0: NoAlias
// CHECK-DAG: pd.fir#0 <-> pd.box.fir#0: MayAlias
// CHECK-DAG: xs(1)%p.fir#0 <-> xs(1)%p.box.fir#0: MayAlias
// CHECK-DAG: s.fir#0 <-> s.addr.fir#0: NoAlias

func.func @_QMmPtest_arrays(%arg0: !fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>> {fir.bindc_name = "arr"}, %arg1: !fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>> {fir.bindc_name = "pol"}, %arg2: !fir.class<!fir.type<_QMmTbase{i:i32}>> {fir.bindc_name = "b"}, %arg3: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>> {fir.bindc_name = "pd"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = fir.dummy_scope : !fir.dscope

  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFtest_arraysEarr") {test.ptr = "arr"} : (!fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>)
  %2 = hlfir.designate %1#0 (%c1) : (!fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  %3 = hlfir.designate %2{"a"}   fortran_attrs<allocatable> {test.ptr = "arr(1)%a"} : (!fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %4:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFtest_arraysEpol") {test.ptr = "pol"} : (!fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>, !fir.dscope) -> (!fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>, !fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>)
  %5 = hlfir.designate %4#0{"a"}   fortran_attrs<allocatable> {test.ptr = "pol%a"} : (!fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %6 = fir.alloca !fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>> {bindc_name = "loc", uniq_name = "_QMmFtest_arraysEloc"}
  %7 = fir.shape %c2 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %6(%7) uniq_name("_QMmFtest_arraysEloc") {test.ptr = "loc"} : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>)
  %9 = hlfir.designate %8#0 (%c1) : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  %10 = hlfir.designate %9{"a"}   fortran_attrs<allocatable> {test.ptr = "loc(1)%a"} : (!fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %11:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QMmFtest_arraysEb") {test.ptr = "b"} : (!fir.class<!fir.type<_QMmTbase{i:i32}>>, !fir.dscope) -> (!fir.class<!fir.type<_QMmTbase{i:i32}>>, !fir.class<!fir.type<_QMmTbase{i:i32}>>)
  %12 = fir.box_addr %11#1 : (!fir.class<!fir.type<_QMmTbase{i:i32}>>) -> !fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>
  %13:2 = hlfir.declare %12 uniq_name("_QMmFtest_arraysEb") : (!fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>) -> (!fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>)
  %14 = hlfir.designate %13#0{"a"}   fortran_attrs<allocatable> {test.ptr = "b%a"} : (!fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %15 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>> {bindc_name = "w", uniq_name = "_QMmFtest_arraysEw"}
  %16:2 = hlfir.declare %15 uniq_name("_QMmFtest_arraysEw") fortran_attrs<allocatable> {test.ptr = "w"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>)
  %17 = fir.load %16#0 {test.ptr = "w.box"} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>

  %18 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>> {bindc_name = "v", uniq_name = "_QMmFtest_arraysEv"}
  %19:2 = hlfir.declare %18 uniq_name("_QMmFtest_arraysEv") fortran_attrs<allocatable> {test.ptr = "v"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>)
  %20 = fir.load %19#0 {test.ptr = "v.box"} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>

  %21 = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>> {bindc_name = "q", uniq_name = "_QMmFtest_arraysEq"}
  %22:2 = hlfir.declare %21 uniq_name("_QMmFtest_arraysEq") fortran_attrs<pointer> {test.ptr = "q"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>)
  %23 = fir.load %22#0 {test.ptr = "q.box"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>

  %24:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 uniq_name("_QMmFtest_arraysEpd") fortran_attrs<pointer> {test.ptr = "pd"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>)
  %25 = fir.load %24#0 {test.ptr = "pd.box"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>

  %26 = fir.alloca !fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>> {bindc_name = "xs", fir.target, uniq_name = "_QMmFtest_arraysExs"}
  %27:2 = hlfir.declare %26(%7) uniq_name("_QMmFtest_arraysExs") fortran_attrs<target> : (!fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>, !fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>)
  %28 = hlfir.designate %27#0 (%c1) : (!fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>, index) -> !fir.ref<!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>
  %29 = hlfir.designate %28{"p"}   fortran_attrs<pointer> {test.ptr = "xs(1)%p"} : (!fir.ref<!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>
  %30 = fir.load %29 {test.ptr = "xs(1)%p.box"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>

  %31 = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>> {bindc_name = "s", uniq_name = "_QMmFtest_arraysEs"}
  %32:2 = hlfir.declare %31 uniq_name("_QMmFtest_arraysEs") fortran_attrs<allocatable> {test.ptr = "s"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>)
  %33 = fir.load %32#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>
  %34 = fir.box_addr %33 {test.ptr = "s.addr"} : (!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>) -> !fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  return
}

func.func @_QMmPtest_arrays.fir(%arg0: !fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>> {fir.bindc_name = "arr"}, %arg1: !fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>> {fir.bindc_name = "pol"}, %arg2: !fir.class<!fir.type<_QMmTbase{i:i32}>> {fir.bindc_name = "b"}, %arg3: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>> {fir.bindc_name = "pd"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = fir.dummy_scope : !fir.dscope

  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFtest_arraysEarr") {test.ptr = "arr.fir"} : (!fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.dscope) -> !fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>
  %2 = fir.array_coor %1 %c1 : (!fir.box<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, index) -> !fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  %3 = fir.coordinate_of %2, a {test.ptr = "arr(1)%a.fir"} : (!fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %4 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFtest_arraysEpol") {test.ptr = "pol.fir"} : (!fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>, !fir.dscope) -> !fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  %5 = fir.coordinate_of %4, a {test.ptr = "pol%a.fir"} : (!fir.class<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %6 = fir.alloca !fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>> {bindc_name = "loc", uniq_name = "_QMmFtest_arraysEloc"}
  %7 = fir.shape %c2 : (index) -> !fir.shape<1>
  %8 = fir.declare %6(%7) uniq_name("_QMmFtest_arraysEloc") {test.ptr = "loc.fir"} : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>
  %9 = fir.array_coor %8(%7) %c1 : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  %10 = fir.coordinate_of %9, a {test.ptr = "loc(1)%a.fir"} : (!fir.ref<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %11 = fir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QMmFtest_arraysEb") {test.ptr = "b.fir"} : (!fir.class<!fir.type<_QMmTbase{i:i32}>>, !fir.dscope) -> !fir.class<!fir.type<_QMmTbase{i:i32}>>
  %12 = fir.box_addr %11 : (!fir.class<!fir.type<_QMmTbase{i:i32}>>) -> !fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>
  %13 = fir.declare %12 uniq_name("_QMmFtest_arraysEb") : (!fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>
  %14 = fir.coordinate_of %13, a {test.ptr = "b%a.fir"} : (!fir.ref<!fir.type<_QMmText{base:!fir.type<_QMmTbase{i:i32}>,a:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %15 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>> {bindc_name = "w", uniq_name = "_QMmFtest_arraysEw"}
  %16 = fir.declare %15 uniq_name("_QMmFtest_arraysEw") fortran_attrs<allocatable> {test.ptr = "w.fir"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>
  %17 = fir.load %16 {test.ptr = "w.box.fir"} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>>

  %18 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>> {bindc_name = "v", uniq_name = "_QMmFtest_arraysEv"}
  %19 = fir.declare %18 uniq_name("_QMmFtest_arraysEv") fortran_attrs<allocatable> {test.ptr = "v.fir"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>
  %20 = fir.load %19 {test.ptr = "v.box.fir"} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTpt{x:i32}>>>>>

  %21 = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>> {bindc_name = "q", uniq_name = "_QMmFtest_arraysEq"}
  %22 = fir.declare %21 uniq_name("_QMmFtest_arraysEq") fortran_attrs<pointer> {test.ptr = "q.fir"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>
  %23 = fir.load %22 {test.ptr = "q.box.fir"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>

  %24 = fir.declare %arg3 dummy_scope %0 arg 4 uniq_name("_QMmFtest_arraysEpd") fortran_attrs<pointer> {test.ptr = "pd.fir"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>
  %25 = fir.load %24 {test.ptr = "pd.box.fir"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>

  %26 = fir.alloca !fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>> {bindc_name = "xs", fir.target, uniq_name = "_QMmFtest_arraysExs"}
  %27 = fir.declare %26(%7) uniq_name("_QMmFtest_arraysExs") fortran_attrs<target> : (!fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>
  %28 = fir.array_coor %27(%7) %c1 : (!fir.ref<!fir.array<2x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>
  %29 = fir.coordinate_of %28, p {test.ptr = "xs(1)%p.fir"} : (!fir.ref<!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>
  %30 = fir.load %29 {test.ptr = "xs(1)%p.box.fir"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp{p:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTtp>>>>}>>>>>

  %31 = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>> {bindc_name = "s", uniq_name = "_QMmFtest_arraysEs"}
  %32 = fir.declare %31 uniq_name("_QMmFtest_arraysEs") fortran_attrs<allocatable> {test.ptr = "s.fir"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>
  %33 = fir.load %32 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>>
  %34 = fir.box_addr %33 {test.ptr = "s.addr.fir"} : (!fir.box<!fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>>) -> !fir.heap<!fir.type<_QMmTt{a:!fir.box<!fir.heap<i32>>}>>
  return
}

// -----

// Pointer dummy arg and arrays of composites.

// The actual argument of a pointer dummy arg (argp) may be another pointer
// dummy arg (argq) or part of a global array of composites with a pointer
// component (garr), of a dummy array of them with TARGET (tarr), or of the
// target of a pointer dummy arg (parr).  argp does not alias a local array
// (lar) or a dummy array with neither TARGET nor POINTER (arr), as only arr may
// change arr(1)%p, and if it does, only arr may reference it (F2023 15.5.2.14).

// module m
//   type t
//     integer, pointer :: p
//     integer :: i
//   end type
//   type(t) :: garr(2)
// contains
//   subroutine test_ptr_arrays(argp, arr, argq, tarr, parr)
//     integer, pointer :: argp, argq
//     type(t) :: arr(:)
//     type(t), target :: tarr(:)
//     type(t), pointer :: parr(:)
//     type(t) :: lar(2)
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest_ptr_arrays"
// CHECK-DAG: argp#0 <-> arr#0: NoAlias
// CHECK-DAG: argp#0 <-> argq#0: MayAlias
// CHECK-DAG: argp#0 <-> tarr#0: MayAlias
// CHECK-DAG: argp#0 <-> parr.tgt#0: MayAlias
// CHECK-DAG: argp#0 <-> garr#0: MayAlias
// CHECK-DAG: argp#0 <-> lar#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arr.fir#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> argq.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> tarr.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> parr.tgt.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> garr.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> lar.fir#0: NoAlias

fir.global @_QMmEgarr : !fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>

func.func @_QMmPtest_ptr_arrays(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg1: !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>> {fir.bindc_name = "arr"}, %arg2: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argq"}, %arg3: !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>> {fir.bindc_name = "tarr", fir.target}, %arg4: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>> {fir.bindc_name = "parr"}) {
  %c2 = arith.constant 2 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFtest_ptr_arraysEargp") fortran_attrs<pointer> {test.ptr = "argp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFtest_ptr_arraysEarr") {test.ptr = "arr"} : (!fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>)
  %3 = fir.shape %c2 : (index) -> !fir.shape<1>
  %4 = fir.address_of(@_QMmEgarr) : !fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  %5:2 = hlfir.declare %4(%3) uniq_name("_QMmEgarr") {test.ptr = "garr"} : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>)
  %6 = fir.alloca !fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {bindc_name = "lar", uniq_name = "_QMmFtest_ptr_arraysElar"}
  %7:2 = hlfir.declare %6(%3) uniq_name("_QMmFtest_ptr_arraysElar") {test.ptr = "lar"} : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>)
  %8:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QMmFtest_ptr_arraysEargq") fortran_attrs<pointer> {test.ptr = "argq"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %9:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 uniq_name("_QMmFtest_ptr_arraysEtarr") fortran_attrs<target> {test.ptr = "tarr"} : (!fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>)
  %10:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 uniq_name("_QMmFtest_ptr_arraysEparr") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>)
  %11 = fir.load %10#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>
  %12 = fir.box_addr %11 {test.ptr = "parr.tgt"} : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>) -> !fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  return
}

func.func @_QMmPtest_ptr_arrays.fir(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg1: !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>> {fir.bindc_name = "arr"}, %arg2: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argq"}, %arg3: !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>> {fir.bindc_name = "tarr", fir.target}, %arg4: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>> {fir.bindc_name = "parr"}) {
  %c2 = arith.constant 2 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFtest_ptr_arraysEargp") fortran_attrs<pointer> {test.ptr = "argp.fir"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFtest_ptr_arraysEarr") {test.ptr = "arr.fir"} : (!fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.dscope) -> !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  %3 = fir.shape %c2 : (index) -> !fir.shape<1>
  %4 = fir.address_of(@_QMmEgarr) : !fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  %5 = fir.declare %4(%3) uniq_name("_QMmEgarr") {test.ptr = "garr.fir"} : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  %6 = fir.alloca !fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {bindc_name = "lar", uniq_name = "_QMmFtest_ptr_arraysElar"}
  %7 = fir.declare %6(%3) uniq_name("_QMmFtest_ptr_arraysElar") {test.ptr = "lar.fir"} : (!fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  %8 = fir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QMmFtest_ptr_arraysEargq") fortran_attrs<pointer> {test.ptr = "argq.fir"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %9 = fir.declare %arg3 dummy_scope %0 arg 4 uniq_name("_QMmFtest_ptr_arraysEtarr") fortran_attrs<target> {test.ptr = "tarr.fir"} : (!fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>, !fir.dscope) -> !fir.box<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  %10 = fir.declare %arg4 dummy_scope %0 arg 5 uniq_name("_QMmFtest_ptr_arraysEparr") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>
  %11 = fir.load %10 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>>
  %12 = fir.box_addr %11 {test.ptr = "parr.tgt.fir"} : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>>) -> !fir.ptr<!fir.array<?x!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>>
  return
}

// -----

// Access paths merged at a region branch miss their start.

// joined is the descriptor of x%c or x%d, so it is part of the data of x
// (xdata).  joined.c is the descriptor of x%c%c or x%d%c, and x%c%c is part of
// the data of x%c (cdata).  The data of x%c does not contain its own
// descriptor (c).  joined2 is c or joined.c: their paths have the same steps,
// but the path of joined.c misses its start.

// CHECK-LABEL: Testing : "_QPtest_merged"
// CHECK-DAG: xdata#0 <-> joined#0: MayAlias
// CHECK-DAG: cdata#0 <-> joined.c#0: MayAlias
// CHECK-DAG: c#0 <-> cdata#0: NoAlias
// CHECK-DAG: cdata#0 <-> joined2#0: MayAlias

!t = !fir.type<_QTt{c:!fir.box<!fir.heap<!fir.type<_QTt>>>,d:!fir.box<!fir.heap<!fir.type<_QTt>>>}>

func.func @_QPtest_merged(%arg0: i1) {
  %0 = fir.alloca !fir.box<!fir.heap<!t>> {uniq_name = "_QFtest_mergedEx"}
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %2 = fir.box_addr %1 {test.ptr = "xdata"} : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %3 = fir.coordinate_of %2, c {test.ptr = "c"} : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %4 = fir.coordinate_of %2, d : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %5 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %6 = fir.box_addr %5 {test.ptr = "cdata"} : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %7 = fir.if %arg0 -> !fir.ref<!fir.box<!fir.heap<!t>>> {
    fir.result %3 : !fir.ref<!fir.box<!fir.heap<!t>>>
  } else {
    fir.result %4 : !fir.ref<!fir.box<!fir.heap<!t>>>
  } {test.ptr = "joined"}
  %8 = fir.load %7 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %9 = fir.box_addr %8 : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %10 = fir.coordinate_of %9, c {test.ptr = "joined.c"} : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %11 = fir.if %arg0 -> !fir.ref<!fir.box<!fir.heap<!t>>> {
    fir.result %3 : !fir.ref<!fir.box<!fir.heap<!t>>>
  } else {
    fir.result %10 : !fir.ref<!fir.box<!fir.heap<!t>>>
  } {test.ptr = "joined2"}
  return
}

// -----

// An access path merged at a region branch may miss a pointer dereference.

// The data of x%p1%a (a1data) is reached through a pointer, so it may contain
// the descriptor of the allocatable dummy arg x, e.g. when x is v%a%q and
// x%p1 => v.  seldata is the data of x%p1%a or x%p2%a.

// CHECK-LABEL: Testing : "_QPtest_merged_ptr"
// CHECK-DAG: x#0 <-> a1data#0: MayAlias
// CHECK-DAG: x#0 <-> seldata#0: MayAlias

!w = !fir.type<_QTw{q:!fir.box<!fir.heap<!fir.type<_QTt>>>}>
!u = !fir.type<_QTu{a:!fir.box<!fir.heap<!w>>}>
!t = !fir.type<_QTt{p1:!fir.box<!fir.ptr<!u>>,p2:!fir.box<!fir.ptr<!u>>}>

func.func @_QPtest_merged_ptr(%arg0: i1, %arg1: !fir.ref<!fir.box<!fir.heap<!t>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 arg 1 uniq_name("_QFtest_merged_ptrEx") fortran_attrs<allocatable> {test.ptr = "x"} : (!fir.ref<!fir.box<!fir.heap<!t>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %4 = fir.coordinate_of %3, p1 : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %5 = fir.coordinate_of %3, p2 : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %6 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %7 = fir.load %5 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %8 = fir.box_addr %6 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %9 = fir.box_addr %7 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %10 = fir.coordinate_of %8, a : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!w>>>
  %11 = fir.coordinate_of %9, a : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!w>>>
  %12 = fir.load %10 : !fir.ref<!fir.box<!fir.heap<!w>>>
  %13 = fir.box_addr %12 {test.ptr = "a1data"} : (!fir.box<!fir.heap<!w>>) -> !fir.heap<!w>
  %14 = fir.if %arg0 -> !fir.ref<!fir.box<!fir.heap<!w>>> {
    fir.result %10 : !fir.ref<!fir.box<!fir.heap<!w>>>
  } else {
    fir.result %11 : !fir.ref<!fir.box<!fir.heap<!w>>>
  }
  %15 = fir.load %14 : !fir.ref<!fir.box<!fir.heap<!w>>>
  %16 = fir.box_addr %15 {test.ptr = "seldata"} : (!fir.box<!fir.heap<!w>>) -> !fir.heap<!w>
  return
}

// -----

// arg%p1%a%z (a1z) is reached through a pointer, so it may be the actual
// argument of the pointer dummy arg argp, even though arg has neither TARGET
// nor POINTER.  selz is arg%p1%a%z or arg%p2%a%z.

// CHECK-LABEL: Testing : "_QPtest_merged_arg"
// CHECK-DAG: argp#0 <-> a1z#0: MayAlias
// CHECK-DAG: argp#0 <-> selz#0: MayAlias

!w = !fir.type<_QTw{z:!fir.box<!fir.ptr<i32>>}>
!u = !fir.type<_QTu{a:!fir.box<!fir.heap<!w>>}>
!t = !fir.type<_QTt{p1:!fir.box<!fir.ptr<!u>>,p2:!fir.box<!fir.ptr<!u>>}>

func.func @_QPtest_merged_arg(%arg0: i1, %arg1: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg2: !fir.ref<!t> {fir.bindc_name = "arg"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 arg 1 uniq_name("_QFtest_merged_argEargp") fortran_attrs<pointer> {test.ptr = "argp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %2 = fir.declare %arg2 dummy_scope %0 arg 2 uniq_name("_QFtest_merged_argEarg") : (!fir.ref<!t>, !fir.dscope) -> !fir.ref<!t>
  %3 = fir.coordinate_of %2, p1 : (!fir.ref<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %4 = fir.coordinate_of %2, p2 : (!fir.ref<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %5 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %6 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %7 = fir.box_addr %5 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %8 = fir.box_addr %6 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %9 = fir.coordinate_of %7, a : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!w>>>
  %10 = fir.coordinate_of %8, a : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!w>>>
  %11 = fir.load %9 : !fir.ref<!fir.box<!fir.heap<!w>>>
  %12 = fir.box_addr %11 : (!fir.box<!fir.heap<!w>>) -> !fir.heap<!w>
  %13 = fir.coordinate_of %12, z {test.ptr = "a1z"} : (!fir.heap<!w>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %14 = fir.if %arg0 -> !fir.ref<!fir.box<!fir.heap<!w>>> {
    fir.result %9 : !fir.ref<!fir.box<!fir.heap<!w>>>
  } else {
    fir.result %10 : !fir.ref<!fir.box<!fir.heap<!w>>>
  }
  %15 = fir.load %14 : !fir.ref<!fir.box<!fir.heap<!w>>>
  %16 = fir.box_addr %15 : (!fir.box<!fir.heap<!w>>) -> !fir.heap<!w>
  %17 = fir.coordinate_of %16, z {test.ptr = "selz"} : (!fir.heap<!w>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  return
}

// -----

// The steps of an access path merged at a region branch do not tell which
// object its components belong to.

// sela is x%p1%q%a or x%p2%q%a, so it may be x%p1%q%a (a1).  The data of x%p1%q
// is part of a pointer target, so x%r may point to its component a.

// CHECK-LABEL: Testing : "_QPtest_merged_diverge"
// CHECK-DAG: a1#0 <-> sela#0: MayAlias
// CHECK-DAG: rtarget#0 <-> a1#0: MayAlias
// CHECK-DAG: rtarget#0 <-> sela#0: MayAlias

!u = !fir.type<_QTu{q:!fir.box<!fir.heap<!fir.type<_QTt>>>}>
!t = !fir.type<_QTt{a:i32,r:!fir.box<!fir.ptr<i32>>,p1:!fir.box<!fir.ptr<!u>>,p2:!fir.box<!fir.ptr<!u>>}>

func.func @_QPtest_merged_diverge(%arg0: i1, %arg1: !fir.ref<!fir.box<!fir.heap<!t>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 arg 1 uniq_name("_QFtest_merged_divergeEx") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!t>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %4 = fir.coordinate_of %3, r : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %5 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %6 = fir.box_addr %5 {test.ptr = "rtarget"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  %7 = fir.coordinate_of %3, p1 : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %8 = fir.coordinate_of %3, p2 : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %9 = fir.load %7 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %10 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %11 = fir.box_addr %9 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %12 = fir.box_addr %10 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %13 = fir.coordinate_of %11, q : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %14 = fir.coordinate_of %12, q : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %15 = fir.load %13 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %16 = fir.box_addr %15 : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %17 = fir.coordinate_of %16, a {test.ptr = "a1"} : (!fir.heap<!t>) -> !fir.ref<i32>
  %18 = fir.if %arg0 -> !fir.ref<!fir.box<!fir.heap<!t>>> {
    fir.result %13 : !fir.ref<!fir.box<!fir.heap<!t>>>
  } else {
    fir.result %14 : !fir.ref<!fir.box<!fir.heap<!t>>>
  }
  %19 = fir.load %18 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %20 = fir.box_addr %19 : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %21 = fir.coordinate_of %20, a {test.ptr = "sela"} : (!fir.heap<!t>) -> !fir.ref<i32>
  return
}

// -----

// The data of x%p1%q (q1data) is part of a pointer target, so it may contain
// the descriptor of the TARGET allocatable dummy arg y: the actual argument may
// be x%p1%q%c.  seldata is the data of x%p1%q or x%p2%q.

// CHECK-LABEL: Testing : "_QPtest_merged_target"
// CHECK-DAG: y#0 <-> q1data#0: MayAlias
// CHECK-DAG: y#0 <-> seldata#0: MayAlias

!u = !fir.type<_QTu{q:!fir.box<!fir.heap<!fir.type<_QTt>>>}>
!t = !fir.type<_QTt{c:!fir.box<!fir.heap<i32>>,p1:!fir.box<!fir.ptr<!u>>,p2:!fir.box<!fir.ptr<!u>>}>

func.func @_QPtest_merged_target(%arg0: i1, %arg1: !fir.ref<!fir.box<!fir.heap<!t>>> {fir.bindc_name = "x"}, %arg2: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "y", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 arg 1 uniq_name("_QFtest_merged_targetEx") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!t>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %2 = fir.declare %arg2 dummy_scope %0 arg 2 uniq_name("_QFtest_merged_targetEy") fortran_attrs<allocatable, target> {test.ptr = "y"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %5 = fir.coordinate_of %4, p1 : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %6 = fir.coordinate_of %4, p2 : (!fir.heap<!t>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %7 = fir.load %5 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %8 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %9 = fir.box_addr %7 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %10 = fir.box_addr %8 : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %11 = fir.coordinate_of %9, q : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %12 = fir.coordinate_of %10, q : (!fir.ptr<!u>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %13 = fir.load %11 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %14 = fir.box_addr %13 {test.ptr = "q1data"} : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %15 = fir.if %arg0 -> !fir.ref<!fir.box<!fir.heap<!t>>> {
    fir.result %11 : !fir.ref<!fir.box<!fir.heap<!t>>>
  } else {
    fir.result %12 : !fir.ref<!fir.box<!fir.heap<!t>>>
  }
  %16 = fir.load %15 : !fir.ref<!fir.box<!fir.heap<!t>>>
  %17 = fir.box_addr %16 {test.ptr = "seldata"} : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  return
}
