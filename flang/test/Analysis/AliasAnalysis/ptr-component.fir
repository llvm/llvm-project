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
  %1:2 = hlfir.declare %0 {uniq_name = "_QMmFfooEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QMmFfooEi2"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QMmFfooEi2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg0 {uniq_name = "_QMmFfooEx", test.ptr = "x"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %5:2 = hlfir.declare %arg1 {uniq_name = "_QMmFfooEy", test.ptr = "y"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %6 = hlfir.designate %4#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "xnext1"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %9 = hlfir.designate %8{"i"} : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  hlfir.assign %10 to %1#0 : i32, !fir.ref<i32>
  hlfir.assign %5#0 to %4#0 : !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %11 = hlfir.designate %4#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "xnext2"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %12 = fir.load %11 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %13 = fir.box_addr %12 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %14 = hlfir.designate %13{"i"} : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %15 = fir.load %14 : !fir.ref<i32>
  hlfir.assign %15 to %3#0 : i32, !fir.ref<i32>
  %16 = hlfir.designate %5#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "ynext"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  return
}

func.func @_QMmPfoo.fir(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %0 = fir.alloca !fir.box<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %1 = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFfooEi1"}
  %2 = fir.declare %1 {uniq_name = "_QMmFfooEi1"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QMmFfooEi2"}
  %4 = fir.declare %3 {uniq_name = "_QMmFfooEi2"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg0 {test.ptr = "x.fir", uniq_name = "_QMmFfooEx"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %6 = fir.declare %arg1 {test.ptr = "y.fir", uniq_name = "_QMmFfooEy"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %7 = fir.field_index next, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %8 = fir.coordinate_of %5, %7 {test.ptr="xnext1.fir"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %9 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %10 = fir.box_addr %9 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %11 = fir.field_index i, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %12 = fir.coordinate_of %10, %11 : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<i32>
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
  %20 = fir.call @_FortranAAssign(%17, %18, %19, %c14_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  %21 = fir.field_index next, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %22 = fir.coordinate_of %5, %21 {test.ptr="xnext2.fir"}: (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %23 = fir.load %22 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %24 = fir.box_addr %23 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %25 = fir.field_index i, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %26 = fir.coordinate_of %24, %25 : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<i32>
  %27 = fir.load %26 : !fir.ref<i32>
  fir.store %27 to %4 : !fir.ref<i32>
  %28 = fir.field_index next, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %29 = fir.coordinate_of %6, %28 {test.ptr="ynext.fir"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  return
}

// -----

// Same test as above focusing on aliasing between x%next and y%next data
// CHECK-LABEL: Testing : "_QMmPfoo2"
// CHECK-DAG: xnext#0 <-> ynext#0: MayAlias
// CHECK-DAG: xnext.fir#0 <-> ynext.fir#0: MayAlias

func.func @_QMmPfoo2(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %4:2 = hlfir.declare %arg0 {uniq_name = "_QMmFfooEx"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %5:2 = hlfir.declare %arg1 {uniq_name = "_QMmFfooEy"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %6 = hlfir.designate %4#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %8 = fir.box_addr %7 {test.ptr = "xnext"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>

  %9 = hlfir.designate %5#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %11 = fir.box_addr %10 {test.ptr = "ynext"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  return
}

func.func @_QMmPfoo2.fir(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %0 = fir.declare %arg0 {uniq_name = "_QMmFfooEx"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %1 = fir.declare %arg1 {uniq_name = "_QMmFfooEy"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %2 = fir.field_index next, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %3 = fir.coordinate_of %0, %2 : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %5 = fir.box_addr %4 {test.ptr = "xnext.fir"} : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %6 = fir.field_index next, !fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>
  %7 = fir.coordinate_of %1, %6 : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
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
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QMmFfooEx"} : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QMmFfooEy"} : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>)
  %2 = hlfir.designate %1#0{"array"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %3 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %4 = hlfir.designate %3 (%c1) {test.ptr = "yarray"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  %5 = fir.load %4 : !fir.ref<i32>
  %6 = hlfir.designate %0#0{"array"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1_0 = arith.constant 1 : index
  %8 = hlfir.designate %7 (%c1_0)  {test.ptr = "xarray"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  hlfir.assign %5 to %8 : i32, !fir.ref<i32>
  return
}

func.func @_QMmPfoo3.fir(%arg0: !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>> {fir.bindc_name = "y"}) {
  %0 = fir.declare %arg0 {uniq_name = "_QMmFfooEx"} : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
  %1 = fir.declare %arg1 {uniq_name = "_QMmFfooEy"} : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
  %2 = fir.field_index array, !fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
  %3 = fir.coordinate_of %1, %2 : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
  %6 = fir.shift %5#0 : (index) -> !fir.shift<1>
  %7 = fir.array_coor %4(%6) %c1 {test.ptr="yarray.fir"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
  %8 = fir.load %7 : !fir.ref<i32>
  %9 = fir.field_index array, !fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
  %10 = fir.coordinate_of %0, %9 : (!fir.ref<!fir.type<_QMmTta{array:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
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
  %1:2 = hlfir.declare %0 {test.ptr="x", uniq_name = "_QMmEx"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>)
  %2 = hlfir.designate %1#0{"p"}   {test.ptr="x%p", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  return
}

func.func @_QMmPtest.fir() {
  %0 = fir.address_of(@_QMmEx) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>
  %1 = fir.declare %0 {test.ptr = "x.fir", uniq_name = "_QMmEx"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>
  %2 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>
  %3 = fir.coordinate_of %1, %2 {test.ptr="x%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<i32>>>
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
//
// TODO: Thus, all results below should be NoAlias.  However, AliasAnalysis
// normally does not recognize when two values (x%x vs. x%i) are distinct
// components of the same composite (x) as opposed to being potentially
// overlapping parts of something, so it returns MayAlias.  There is special
// handling for a pointer component (x%p) that does recognize it as separate
// from other components (x%i).  But it does not yet distinguish the composite
// (x) from a component (x%x) that is also a composite with a pointer component
// (x%x%p).  Thus, because x and x%p can alias, it assumes x%x and x%p can too.
// CHECK-DAG: x%x#0 <-> x%i#0: MayAlias
// CHECK-DAG: x%p#0 <-> x%i#0: NoAlias
// CHECK-DAG: x%x#0 <-> x%p#0: MayAlias
// CHECK-DAG: x%x.fir#0 <-> x%i.fir#0: MayAlias
// CHECK-DAG: x%p.fir#0 <-> x%i.fir#0: NoAlias
// CHECK-DAG: x%x.fir#0 <-> x%p.fir#0: MayAlias

func.func @_QMmPtest() {
  %0 = fir.alloca !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "x", uniq_name = "_QMmFtestEx"}
  %1:2 = hlfir.declare %0 {test.ptr="x", uniq_name = "_QMmFtestEx"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %2 = hlfir.designate %1#0{"x"}  {test.ptr="x%x"}  : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>
  %3 = hlfir.designate %1#0{"p"}   {test.ptr="x%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %4 = hlfir.designate %1#0{"i"} {test.ptr="x%i"}  : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>
  %5 = hlfir.designate %2{"p"}   {test.ptr="x%x%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  return
}

func.func @_QMmPtest.fir() {
  %0 = fir.alloca !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "x", uniq_name = "_QMmFtestEx"}
  %1 = fir.declare %0 {test.ptr = "x.fir", uniq_name = "_QMmFtestEx"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %2 = fir.field_index x, !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %3 = fir.coordinate_of %1, %2 {test.ptr="x%x.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>
  %4 = fir.field_index p, !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %5 = fir.coordinate_of %1, %4 {test.ptr="x%p.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %6 = fir.field_index i, !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %7 = fir.coordinate_of %1, %6 {test.ptr="x%i.fir"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>
  %8 = fir.field_index p, !fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>
  %9 = fir.coordinate_of %3, %8 {test.ptr="x%x%p.fir"} : (!fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
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
// CHECK-DAG: argp#0 <-> arg#0: MayAlias
// CHECK-DAG: argp#0 <-> arg%p#0: MayAlias
// CHECK-DAG: argp#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> arg%p.fir#0: MayAlias
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

  %1:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="argp", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFtestEargp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.box_addr %2 {test.ptr="argp.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %4:2 = hlfir.declare %arg1 dummy_scope %0 {test.ptr="arga", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmFtestEarga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)

  %5:2 = hlfir.declare %arg2 dummy_scope %0 {test.ptr="arg", uniq_name = "_QMmFtestEarg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %6 = hlfir.designate %5#0{"p"}   {test.ptr="arg%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %7 = hlfir.designate %5#0{"i"} {test.ptr="arg%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %8 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %9:2 = hlfir.declare %8 {test.ptr="glob", uniq_name = "_QMmEglob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %10 = hlfir.designate %9#0{"p"}   {test.ptr="glob%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %11 = hlfir.designate %9#0{"i"} {test.ptr="glob%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %12 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFtestEloc"}
  %13:2 = hlfir.declare %12 {test.ptr="loc", uniq_name = "_QMmFtestEloc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %14 = hlfir.designate %13#0{"p"}   {test.ptr="loc%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %15 = hlfir.designate %13#0{"i"} {test.ptr="loc%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}

func.func @_QMmPtest.fir(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "arga"}, %arg2: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}) {
  %0 = fir.dummy_scope : !fir.dscope

  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "argp.fir", uniq_name = "_QMmFtestEargp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.box_addr %2 {test.ptr = "argp.tgt.fir"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %4 = fir.declare %arg1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, test.ptr = "arga.fir", uniq_name = "_QMmFtestEarga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %5 = fir.declare %arg2 dummy_scope %0 {test.ptr = "arg.fir", uniq_name = "_QMmFtestEarg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %6 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %7 = fir.coordinate_of %5, %6 {test.ptr="arg%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %8 = fir.field_index i, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %9 = fir.coordinate_of %5, %8 {test.ptr="arg%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>

  %10 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %11 = fir.declare %10 {test.ptr = "glob.fir", uniq_name = "_QMmEglob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %12 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %13 = fir.coordinate_of %11, %12 {test.ptr="glob%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %14 = fir.field_index i, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %15 = fir.coordinate_of %11, %14 {test.ptr="glob%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>

  %16 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFtestEloc"}
  %17 = fir.declare %16 {test.ptr = "loc.fir", uniq_name = "_QMmFtestEloc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %18 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %19 = fir.coordinate_of %17, %18 {test.ptr="loc%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %20 = fir.field_index i, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %21 = fir.coordinate_of %17, %20 {test.ptr="loc%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>

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
// CHECK-DAG: argp#0 <-> arg#0: MayAlias
// CHECK-DAG: argp#0 <-> arg%p#0: MayAlias
// CHECK-DAG: argp#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.fir#0 <-> arg.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> arg%p.fir#0: MayAlias
// CHECK-DAG: argp.fir#0 <-> arg%i.fir#0: NoAlias
//
// TODO: Shouldn't these be NoAlias?  However, argp.tgt is currently handled as
// Indirect.
// CHECK-DAG: argp.tgt#0 <-> arg#0: MayAlias
// CHECK-DAG: argp.tgt#0 <-> arg%p#0: MayAlias
// CHECK-DAG: argp.tgt#0 <-> arg%i#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg.fir#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg%p.fir#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> arg%i.fir#0: MayAlias
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
// TODO: Shouldn't these be NoAlias?  However, argp.tgt is currently handled as
// Indirect.
// CHECK-DAG: argp.tgt#0 <-> glob#0: MayAlias
// CHECK-DAG: argp.tgt#0 <-> glob%p#0: MayAlias
// CHECK-DAG: argp.tgt#0 <-> glob%i#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob.fir#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob%p.fir#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> glob%i.fir#0: MayAlias
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
// TODO: Shouldn't these be NoAlias?  However, argp.tgt is currently handled as
// Indirect.
// CHECK-DAG: argp.tgt#0 <-> loc#0: MayAlias
// CHECK-DAG: argp.tgt#0 <-> loc%p#0: MayAlias
// CHECK-DAG: argp.tgt#0 <-> loc%i#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc.fir#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc%p.fir#0: MayAlias
// CHECK-DAG: argp.tgt.fir#0 <-> loc%i.fir#0: MayAlias
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
  %3:2 = hlfir.declare %2 {test.ptr="argp", fortran_attrs = #fir.var_attrs<pointer, host_assoc>, uniq_name = "_QMmFparentEargp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %4 = fir.load %3#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %5 = fir.box_addr %4 {test.ptr="argp.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %c1_i32 = arith.constant 1 : i32
  %10 = fir.coordinate_of %arg1, %c1_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %11 = fir.load %10 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %12:2 = hlfir.declare %11 {test.ptr="arga", fortran_attrs = #fir.var_attrs<allocatable, host_assoc>, uniq_name = "_QMmFparentEarga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)

  %20 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %21:2 = hlfir.declare %20 {test.ptr="glob", uniq_name = "_QMmEglob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %22 = hlfir.designate %21#0{"p"}   {test.ptr="glob%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %23 = hlfir.designate %21#0{"i"} {test.ptr="glob%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %30:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="arg", uniq_name = "_QMmFparentFtestEarg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %31 = hlfir.designate %30#0{"p"}   {test.ptr="arg%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %32 = hlfir.designate %30#0{"i"} {test.ptr="arg%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %40 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFparentFtestEloc"}
  %41:2 = hlfir.declare %40 {test.ptr="loc", uniq_name = "_QMmFparentFtestEloc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %42 = hlfir.designate %41#0{"p"}   {test.ptr="loc%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %43 = hlfir.designate %41#0{"i"} {test.ptr="loc%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}

func.func private @_QMmFparentPtest.fir(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}, %arg1: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QMmPparent, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope

  %c0_i32 = arith.constant 0 : i32
  %1 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>
  %2 = fir.load %1 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>
  %3 = fir.declare %2 {fortran_attrs = #fir.var_attrs<pointer, host_assoc>, test.ptr = "argp.fir", uniq_name = "_QMmFparentEargp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %5 = fir.box_addr %4 {test.ptr = "argp.tgt.fir"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %c1_i32 = arith.constant 1 : i32
  %6 = fir.coordinate_of %arg1, %c1_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %7 = fir.load %6 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<i32>>>>
  %8 = fir.declare %7 {fortran_attrs = #fir.var_attrs<allocatable, host_assoc>, test.ptr = "arga.fir", uniq_name = "_QMmFparentEarga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>

  %9 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %10 = fir.declare %9 {test.ptr = "glob.fir", uniq_name = "_QMmEglob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %11 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %12 = fir.coordinate_of %10, %11 {test.ptr="glob%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %13 = fir.field_index i, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %14 = fir.coordinate_of %10, %13 {test.ptr="glob%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>

  %15 = fir.declare %arg0 dummy_scope %0 {test.ptr = "arg.fir", uniq_name = "_QMmFparentFtestEarg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %16 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %17 = fir.coordinate_of %15, %16 {test.ptr="arg%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %18 = fir.field_index i, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %19 = fir.coordinate_of %15, %18 {test.ptr="arg%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>

  %20 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFparentFtestEloc"}
  %21 = fir.declare %20 {test.ptr = "loc.fir", uniq_name = "_QMmFparentFtestEloc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %22 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %23 = fir.coordinate_of %21, %22 {test.ptr="loc%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %24 = fir.field_index i, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>
  %25 = fir.coordinate_of %21, %24 {test.ptr="loc%i.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.field) -> !fir.ref<i32>

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

// TODO: All glob0 vs. glob1 cases can be NoAlias.  However, AliasAnalysis
// currently indiscriminately treats all targets that are data (addresses of
// glob[01] and glob[01]%r but not glob[01]%p and glob[01]%a) as aliasing.

// Check composite vs. composite.
//
// CHECK-DAG: arg0#0 <-> arg1#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> arg1.fir#0: MayAlias
//
// CHECK-DAG: arg0#0 <-> glob0#0: MayAlias
// CHECK-DAG: arg0.fir#0 <-> glob0.fir#0: MayAlias
//
// CHECK-DAG: glob0#0 <-> glob1#0: MayAlias
// CHECK-DAG: glob0.fir#0 <-> glob1.fir#0: MayAlias

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
// CHECK-DAG: glob0%r#0 <-> glob1#0: MayAlias
// CHECK-DAG: glob0%p.fir#0 <-> glob1.fir#0: NoAlias
// CHECK-DAG: glob0%a.fir#0 <-> glob1.fir#0: NoAlias
// CHECK-DAG: glob0%r.fir#0 <-> glob1.fir#0: MayAlias

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
// CHECK-DAG: glob0#0 <-> glob1%r#0: MayAlias
// CHECK-DAG: glob0.fir#0 <-> glob1%p.fir#0: NoAlias
// CHECK-DAG: glob0.fir#0 <-> glob1%a.fir#0: NoAlias
// CHECK-DAG: glob0.fir#0 <-> glob1%r.fir#0: MayAlias

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
// CHECK-DAG: glob0%r.fir#0 <-> glob1%r.fir#0: MayAlias

func.func @_QMmPtest(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg0", fir.target}, %arg1: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg1", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope

  %10:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="arg0", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmFtestEarg0"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %11 = hlfir.designate %10#0{"p"}   {test.ptr="arg0%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %12 = hlfir.designate %10#0{"a"}   {test.ptr="arg0%a", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %13 = hlfir.designate %10#0{"r"}   {test.ptr="arg0%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %20:2 = hlfir.declare %arg1 dummy_scope %0 {test.ptr="arg1", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmFtestEarg1"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %21 = hlfir.designate %20#0{"p"}   {test.ptr="arg1%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %22 = hlfir.designate %20#0{"a"}   {test.ptr="arg1%a", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %23 = hlfir.designate %20#0{"r"}   {test.ptr="arg1%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %30 = fir.address_of(@_QMmEglob0) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %31:2 = hlfir.declare %30 {test.ptr="glob0", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEglob0"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %32 = hlfir.designate %31#0{"p"}   {test.ptr="glob0%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %33 = hlfir.designate %31#0{"a"}   {test.ptr="glob0%a", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %34 = hlfir.designate %31#0{"r"}   {test.ptr="glob0%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  %40 = fir.address_of(@_QMmEglob1) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %41:2 = hlfir.declare %40 {test.ptr="glob1", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEglob1"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>)
  %42 = hlfir.designate %41#0{"p"}   {test.ptr="glob1%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %43 = hlfir.designate %41#0{"a"}   {test.ptr="glob1%a", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %44 = hlfir.designate %41#0{"r"}   {test.ptr="glob1%r"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<f32>

  return
}

func.func @_QMmPtest.fir(%arg0: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg0", fir.target}, %arg1: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>> {fir.bindc_name = "arg1", fir.target}) {
  %0 = fir.dummy_scope : !fir.dscope

  %1 = fir.declare %arg0 dummy_scope %0 {test.ptr="arg0.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmFtestEarg0"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %2 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %3 = fir.coordinate_of %1, %2 {test.ptr="arg0%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.field_index a, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %5 = fir.coordinate_of %1, %4 {test.ptr="arg0%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %6 = fir.field_index r, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %7 = fir.coordinate_of %1, %6 {test.ptr="arg0%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<f32>

  %8 = fir.declare %arg1 dummy_scope %0 {test.ptr="arg1.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmFtestEarg1"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %9 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %10 = fir.coordinate_of %8, %9 {test.ptr="arg1%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.field_index a, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %12 = fir.coordinate_of %8, %11 {test.ptr="arg1%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %13 = fir.field_index r, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %14 = fir.coordinate_of %8, %13 {test.ptr="arg1%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<f32>

  %15 = fir.address_of(@_QMmEglob0) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %16 = fir.declare %15 {test.ptr="glob0.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEglob0"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %17 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %18 = fir.coordinate_of %16, %17 {test.ptr="glob0%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %19 = fir.field_index a, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %20 = fir.coordinate_of %16, %19 {test.ptr="glob0%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %21 = fir.field_index r, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %22 = fir.coordinate_of %16, %21 {test.ptr="glob0%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<f32>

  %23 = fir.address_of(@_QMmEglob1) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %24 = fir.declare %23 {test.ptr="glob1.fir", fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmEglob1"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>) -> !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>
  %25 = fir.field_index p, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %26 = fir.coordinate_of %24, %25 {test.ptr="glob1%p.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %27 = fir.field_index a, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %28 = fir.coordinate_of %24, %27 {test.ptr="glob1%a.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %29 = fir.field_index r, !fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>
  %30 = fir.coordinate_of %24, %29 {test.ptr="glob1%r.fir"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<f32>>,a:!fir.box<!fir.heap<f32>>,r:f32}>>, !fir.field) -> !fir.ref<f32>

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

  %10:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="arg0", fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmFtestEarg0"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %11:2 = hlfir.declare %arg1 dummy_scope %0 {test.ptr="arg1", fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmFtestEarg1"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  %20 = fir.address_of(@_QMmEglob0) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %21:2 = hlfir.declare %20 {test.ptr="glob0", fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmEglob0"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %22 = fir.address_of(@_QMmEglob1) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %23:2 = hlfir.declare %22 {test.ptr="glob1", fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmEglob1"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  return
}
