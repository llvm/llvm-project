! Test lowering of CSHIFT intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

module types
  type t
  end type t
end module types

! 1d shift by scalar
subroutine cshift1(a, s)
  integer :: a(:), s
  a = CSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift1(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

! 1d shift by scalar with dim
subroutine cshift2(a, s)
  integer :: a(:), s
  a = CSHIFT(a, 2, 1)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift2(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_5]] dim %[[VAL_6]] : (!fir.box<!fir.array<?xi32>>, i32, i32) -> !hlfir.expr<?xi32>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by scalar
subroutine cshift3(a, s)
  integer :: a(:,:), s
  a = CSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift3(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?x?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by scalar with dim
subroutine cshift4(a, s)
  integer :: a(:,:), s
  a = CSHIFT(a, 2, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift4(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_5]] dim %[[VAL_6]] : (!fir.box<!fir.array<?x?xi32>>, i32, i32) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by array
subroutine cshift5(a, s)
  integer :: a(:,:), s(:)
  a = CSHIFT(a, s)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift5(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_4]]#0 : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_5]] : !hlfir.expr<?x?xi32>
! CHECK:           return
! CHECK:         }

! 2d shift by array expr
subroutine cshift6(a, s)
  integer :: a(:,:), s(:)
  a = CSHIFT(a, s + 1)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift6(
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
! CHECK:           %[[VAL_14:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_9]] : (!fir.box<!fir.array<?x?xi32>>, !hlfir.expr<?xi32>) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_3]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<?x?xi32>
! CHECK:           hlfir.destroy %[[VAL_9]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

! 1d character(10,2) shift by scalar
subroutine cshift7(a, s)
  character(10,2) :: a(:)
  a = CSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift7(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<2,10>>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_6:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.cshift %[[VAL_4]]#0 %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<2,10>>>, i32) -> !hlfir.expr<?x!fir.char<2,10>>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_4]]#0 : !hlfir.expr<?x!fir.char<2,10>>, !fir.box<!fir.array<?x!fir.char<2,10>>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x!fir.char<2,10>>
! CHECK:           return
! CHECK:         }

! 1d character(*) shift by scalar
subroutine cshift8(a, s)
  character(*) :: a(:)
  a = CSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift8(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32) -> !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

! 1d type(t) shift by scalar
subroutine cshift9(a, s)
  use types
  type(t) :: a(:)
  a = CSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift9(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "a"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = hlfir.cshift %[[VAL_3]]#0 %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, i32) -> !hlfir.expr<?x!fir.type<_QMtypesTt>>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : !hlfir.expr<?x!fir.type<_QMtypesTt>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
! CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<?x!fir.type<_QMtypesTt>>
! CHECK:           return
! CHECK:         }

! 1d class(t) shift by scalar
subroutine cshift10(a, s)
  use types
  class(t), allocatable :: a(:)
  a = CSHIFT(a, 2)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift10(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>> {fir.bindc_name = "a"},
! CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
! CHECK:           %[[VAL_7:.*]] = hlfir.cshift %[[VAL_6]] %[[VAL_5]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i32) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_3]]#0 realloc : !hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
! CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
! CHECK:           return
! CHECK:         }

! 1d shift by scalar with variable dim
subroutine cshift11(a, s, d)
  integer :: a(:), s, d
  a = CSHIFT(a, 2, d)
end subroutine
! CHECK-LABEL:   func.func @_QPcshift11(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"},
! CHECK-SAME:                           %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFcshift11Ea"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {uniq_name = "_QFcshift11Ed"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_3]] {uniq_name = "_QFcshift11Es"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = hlfir.cshift %[[VAL_4]]#0 %[[VAL_7]] dim %[[VAL_8]] : (!fir.box<!fir.array<?xi32>>, i32, i32) -> !hlfir.expr<?xi32>
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_9]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }
