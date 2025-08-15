! Test lowering of EOSHIFT intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

module eoshift_types
  type t
  end type t
end module eoshift_types

! 1d shift by scalar
subroutine eoshift1(a, s)
  integer :: a(:), s
  a = EOSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift1(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

! 1d shift by scalar with dim
subroutine eoshift2(a, s)
  integer :: a(:), s
  a = EOSHIFT(a, 2, dim=1)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift2(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_5]] dim %[[VAL_6]] : (!fir.box<!fir.array<?xi32>>, i32, i32) -> !hlfir.expr<?xi32>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by scalar
subroutine eoshift3(a, s)
  integer :: a(:,:), s
  a = EOSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift3(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?x?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by scalar with dim
subroutine eoshift4(a, s)
  integer :: a(:,:), s
  a = EOSHIFT(a, 2, dim=2)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift4(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_5]] dim %[[VAL_6]] : (!fir.box<!fir.array<?x?xi32>>, i32, i32) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by array
subroutine eoshift5(a, s)
  integer :: a(:,:), s(:)
  a = EOSHIFT(a, s)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift5(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_4]]#0 : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_5]] : !hlfir.expr<?x?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by array expr
subroutine eoshift6(a, s)
  integer :: a(:,:), s(:)
  a = EOSHIFT(a, s + 1)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift6(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_6]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32>
! CHECK:           %[[VAL_14:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_9]] : (!fir.box<!fir.array<?x?xi32>>, !hlfir.expr<?xi32>) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<?x?xi32>
! CHECK:           hlfir.destroy %[[VAL_9]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

! 1d character(10,2) shift by scalar
subroutine eoshift7(a, s)
  character(10,2) :: a(:)
  a = EOSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift7(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<2,10>>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_6:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.eoshift %[[VAL_4]]#0 %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<2,10>>>, i32) -> !hlfir.expr<?x!fir.char<2,10>>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_4]]#0 : !hlfir.expr<?x!fir.char<2,10>>, !fir.box<!fir.array<?x!fir.char<2,10>>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x!fir.char<2,10>>
! CHECK:           return
! CHECK:         }

! 1d character(*) shift by scalar
subroutine eoshift8(a, s)
  character(*) :: a(:)
  a = EOSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift8(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.eoshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32) -> !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

! 1d type(t) shift by scalar
subroutine eoshift9(a, s)
  use eoshift_types
  type(t) :: a(:)
  a = EOSHIFT(a, 2, boundary=t())
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift9(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>> {fir.bindc_name = "a"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFeoshift9Ea"} : (!fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFeoshift9Es"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQro._QMeoshift_typesTt.0) : !fir.ref<!fir.type<_QMeoshift_typesTt>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QMeoshift_typesTt.0"} : (!fir.ref<!fir.type<_QMeoshift_typesTt>>) -> (!fir.ref<!fir.type<_QMeoshift_typesTt>>, !fir.ref<!fir.type<_QMeoshift_typesTt>>)
! CHECK:           %[[VAL_6:.*]] = hlfir.eoshift %[[VAL_1]]#0 %[[VAL_3]] boundary %[[VAL_5]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>, i32, !fir.ref<!fir.type<_QMeoshift_typesTt>>) -> !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>>, !fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>>
! CHECK:           return
! CHECK:         }

! 1d class(t) shift by scalar
subroutine eoshift10(a, s)
  use eoshift_types
  class(t), allocatable :: a(:)
  a = EOSHIFT(a, 2, boundary=t())
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift10(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>> {fir.bindc_name = "a"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFeoshift10Ea"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFeoshift10Es"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQro._QMeoshift_typesTt.1) : !fir.ref<!fir.type<_QMeoshift_typesTt>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QMeoshift_typesTt.1"} : (!fir.ref<!fir.type<_QMeoshift_typesTt>>) -> (!fir.ref<!fir.type<_QMeoshift_typesTt>>, !fir.ref<!fir.type<_QMeoshift_typesTt>>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>>
! CHECK:           %[[VAL_7:.*]] = hlfir.eoshift %[[VAL_6]] %[[VAL_3]] boundary %[[VAL_5]]#0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>, i32, !fir.ref<!fir.type<_QMeoshift_typesTt>>) -> !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>?>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_1]]#0 realloc : !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>?>
! CHECK:           return
! CHECK:         }

! 1d shift by scalar with variable dim
subroutine eoshift11(a, s, d)
  integer :: a(:), s, d
  a = EOSHIFT(a, 2, dim=d)
end subroutine
! CHECK-LABEL:   func.func @_QPeoshift11(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"},
! CHECK-SAME:                           %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift11Ea"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift11Ed"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift11Es"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = hlfir.eoshift %[[VAL_4]]#0 %[[VAL_7]] dim %[[VAL_8]] : (!fir.box<!fir.array<?xi32>>, i32, i32) -> !hlfir.expr<?xi32>
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_9]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

subroutine eoshift12(array, shift, boundary, dim)
  real :: array(:,:)
  real, optional :: boundary
  integer :: shift(:), dim
  array = EOSHIFT(array, shift, boundary, dim)
end subroutine eoshift12
! CHECK-LABEL:   func.func @_QPeoshift12(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "shift"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<f32> {fir.bindc_name = "boundary", fir.optional},
! CHECK-SAME:      %[[ARG3:.*]]: !fir.ref<i32> {fir.bindc_name = "dim"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFeoshift12Earray"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift12Eboundary"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %[[VAL_0]] {uniq_name = "_QFeoshift12Edim"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFeoshift12Eshift"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_5:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.ref<f32>) -> i1
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_2]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK:           %[[VAL_7:.*]] = fir.absent !fir.box<f32>
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : !fir.box<f32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = hlfir.eoshift %[[VAL_1]]#0 %[[VAL_4]]#0 boundary %[[VAL_8]] dim %[[VAL_9]] : (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?xi32>>, !fir.box<f32>, i32) -> !hlfir.expr<?x?xf32>
! CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
! CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<?x?xf32>
! CHECK:           return
! CHECK:         }

! Test scalar logical boundary.
! CHECK-LABEL:   func.func @_QPeoshift13(
subroutine eoshift13(array)
  logical(1) :: array(:)
  array = EOSHIFT(array, -1, .true._1)
! CHECK:           %[[VAL_5:.*]] = hlfir.eoshift %{{.*}} %{{.*}} boundary %{{.*}} : (!fir.box<!fir.array<?x!fir.logical<1>>>, i32, !fir.logical<1>) -> !hlfir.expr<?x!fir.logical<1>>
  array = EOSHIFT(array.EQV..false., -1, .true.)
! CHECK:           %[[VAL_24:.*]] = hlfir.eoshift %{{.*}} %{{.*}} boundary %{{.*}} : (!hlfir.expr<?x!fir.logical<4>>, i32, !fir.logical<4>) -> !hlfir.expr<?x!fir.logical<4>>
end subroutine eoshift13
