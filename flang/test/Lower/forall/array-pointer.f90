! Test lowering of arrays of POINTER.
!
! An array of pointer to T can be constructed by having an array of
! derived type, where the derived type has a pointer to T
! component. An entity with both the DIMENSION and POINTER attributes
! is a pointer to an array of T and never an array of pointer to T in
! Fortran.

! RUN: bbc -emit-fir %s -o - | FileCheck %s

module array_of_pointer_test
  type t
     integer, POINTER :: ip
  end type t

  type u
     integer :: v
  end type u

  type tu
     type(u), POINTER :: ip
  end type tu

  type ta
     integer, POINTER :: ip(:)
  end type ta

  type tb
     integer, POINTER :: ip(:,:)
  end type tb

  type tv
     type(tu), POINTER :: jp(:)
  end type tv

  ! Derived types with type parameters hit a TODO.
!  type ct(l)
!     integer, len :: l
!     character(LEN=l), POINTER :: cp
!  end type ct

!  type cu(l)
!     integer, len :: l
!     character(LEN=l) :: cv
!  end type cu
end module array_of_pointer_test

subroutine s1(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer :: y(:)

  forall (i=1:10)
     ! assign value to pointee variable
     x(i)%ip = y(i)
  end forall
end subroutine s1

! CHECK-LABEL: func @_QPs1(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_14]] : index
! CHECK:           %[[VAL_19:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_18]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_20]] : index
! CHECK:           %[[VAL_25:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_26:.*]] = fir.array_access %[[VAL_12]], %[[VAL_24]], %[[VAL_25]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, index, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_28:.*]] = fir.box_addr %[[VAL_27]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_28]] : !fir.ptr<i32>
! CHECK:           %[[VAL_29:.*]] = fir.array_amend %[[VAL_12]], %[[VAL_26]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_29]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_30:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s1_1(x,y)
  use array_of_pointer_test
  type(t) :: x(10)
  integer :: y(10)

  forall (i=1:10)
     ! assign value to pointee variable
     x(i)%ip = y(i)
  end forall
end subroutine s1_1

! CHECK-LABEL: func @_QPs1_1(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:                %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) : (!fir.ref<!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_1]](%[[VAL_12]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_9]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_11]]) -> (!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_18]] : index
! CHECK:           %[[VAL_23:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_22]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_24]] : index
! CHECK:           %[[VAL_29:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_30:.*]] = fir.array_access %[[VAL_16]], %[[VAL_28]], %[[VAL_29]] : (!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, index, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_30]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_32:.*]] = fir.box_addr %[[VAL_31]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           fir.store %[[VAL_23]] to %[[VAL_32]] : !fir.ptr<i32>
! CHECK:           %[[VAL_33:.*]] = fir.array_amend %[[VAL_16]], %[[VAL_30]] : (!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_33]] : !fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_34:.*]] to %[[VAL_0]] : !fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.ref<!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

! Dependent type assignment, TODO
!subroutine s1_2(x,y,l)
!  use array_of_pointer_test
!  type(ct(l)) :: x(10)
!  character(l) :: y(10)

!  forall (i=1:10)
     ! assign value to pointee variable
!     x(i)%cp = y(i)
!  end forall
!end subroutine s1_2

subroutine s2(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer, TARGET :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
end subroutine s2

! CHECK-LABEL: func @_QPs2(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y", fir.target}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_14]] : index
! CHECK:           %[[VAL_19:.*]] = fir.array_access %[[VAL_9]], %[[VAL_18]] : (!fir.array<?xi32>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_21:.*]] = fir.embox %[[VAL_20]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_25]], %[[VAL_22]] : index
! CHECK:           %[[VAL_27:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_28:.*]] = fir.array_update %[[VAL_12]], %[[VAL_21]], %[[VAL_26]], %[[VAL_27]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.ptr<i32>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_28]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_29:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s2_1(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer, POINTER :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
end subroutine s2_1

! CHECK-LABEL: func @_QPs2_1(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:                %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_12:.*]] = fir.shift %[[VAL_11]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_9]](%[[VAL_12]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_8]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_11]]#0 : index
! CHECK:           %[[VAL_22:.*]] = fir.array_access %[[VAL_13]], %[[VAL_21]] : (!fir.array<?xi32>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_23]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_28]], %[[VAL_25]] : index
! CHECK:           %[[VAL_30:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_31:.*]] = fir.array_update %[[VAL_16]], %[[VAL_24]], %[[VAL_29]], %[[VAL_30]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.ptr<i32>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_31]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_32:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s2_2(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer, ALLOCATABLE, TARGET :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
end subroutine s2_2

! CHECK-LABEL: func @_QPs2_2(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:                %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "y", fir.target}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_10]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_12:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[VAL_13:.*]] = fir.shape_shift %[[VAL_11]]#0, %[[VAL_11]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_12]](%[[VAL_13]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_8]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_11]]#0 : index
! CHECK:           %[[VAL_23:.*]] = fir.array_access %[[VAL_14]], %[[VAL_22]] : (!fir.array<?xi32>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_25:.*]] = fir.embox %[[VAL_24]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:           %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_31:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_32:.*]] = fir.array_update %[[VAL_17]], %[[VAL_25]], %[[VAL_30]], %[[VAL_31]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.ptr<i32>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_32]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_33:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s2_3(x)
  use array_of_pointer_test
  type(t) :: x(:)
  ! This is legal, but a bad idea.
  integer, ALLOCATABLE, TARGET :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
  ! x's pointers will remain associated, and may point to deallocated y.
end subroutine s2_3

! CHECK-LABEL: func @_QPs2_3(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "y", fir.target, uniq_name = "_QFs2_3Ey"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFs2_3Ey.addr"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFs2_3Ey.lb0"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFs2_3Ey.ext0"}
! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_15:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         %[[VAL_16:.*]] = fir.shape_shift %[[VAL_13]], %[[VAL_14]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_15]](%[[VAL_16]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_12]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
! CHECK:           fir.store %[[VAL_21]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_13]] : index
! CHECK:           %[[VAL_26:.*]] = fir.array_access %[[VAL_17]], %[[VAL_25]] : (!fir.array<?xi32>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_28:.*]] = fir.embox %[[VAL_27]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
! CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_32]], %[[VAL_29]] : index
! CHECK:           %[[VAL_34:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_35:.*]] = fir.array_update %[[VAL_20]], %[[VAL_28]], %[[VAL_33]], %[[VAL_34]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.ptr<i32>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_35]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_36:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

! Dependent type - TODO
!subroutine s2_4(x,y,l)
!  use array_of_pointer_test
!  type(ct(l)) :: x(:)
!  character(l), TARGET :: y(:)

!  forall (i=1:10)
     ! assign address to POINTER
!     x(i)%cp => y(i)
!  end forall
!end subroutine s2_4

subroutine s3(x,y)
  use array_of_pointer_test
  type(tu) :: x(:)
  integer :: y(:)

  forall (i=1:10)
     ! assign value to variable, indirecting through box
     x(i)%ip%v = y(i)
  end forall
end subroutine s3

! CHECK-LABEL: func @_QPs3(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_14]] : index
! CHECK:           %[[VAL_19:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_18]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_20]] : index
! CHECK:           %[[VAL_25:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>
! CHECK:           %[[VAL_26:.*]] = fir.field_index v, !fir.type<_QMarray_of_pointer_testTu{v:i32}>
! CHECK:           %[[VAL_27:.*]] = fir.array_access %[[VAL_12]], %[[VAL_24]], %[[VAL_25]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, index, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>
! CHECK:           %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_28]], %[[VAL_26]] : (!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_29]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.array_amend %[[VAL_12]], %[[VAL_27]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:           fir.result %[[VAL_30]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s3_1(x,y)
  use array_of_pointer_test
  type(tu) :: x(:)
  integer :: y(:)

  forall (i=1:10)
     ! assign value to variable, indirecting through box
     x(i)%ip%v = y(i)
  end forall
end subroutine s3_1

! CHECK-LABEL: func @_QPs3_1(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:                %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_14]] : index
! CHECK:           %[[VAL_19:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_18]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_20]] : index
! CHECK:           %[[VAL_25:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>
! CHECK:           %[[VAL_26:.*]] = fir.field_index v, !fir.type<_QMarray_of_pointer_testTu{v:i32}>
! CHECK:           %[[VAL_27:.*]] = fir.array_access %[[VAL_12]], %[[VAL_24]], %[[VAL_25]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, index, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>
! CHECK:           %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_28]], %[[VAL_26]] : (!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_29]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.array_amend %[[VAL_12]], %[[VAL_27]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:           fir.result %[[VAL_30]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>
! CHECK:         return
! CHECK:       }

! Slice a target array and assign the box to a pointer of rank-1 field.
! RHS is an array section. Hits a TODO.
subroutine s4(x,y)
  use array_of_pointer_test
  type(ta) :: x(:)
  integer, TARGET :: y(:)

  forall (i=1:10)
     ! TODO: auto boxing of ranked RHS
!    x(i)%ip => y(i:i+1)
  end forall
end subroutine s4

! Most other Fortran implementations cannot compile the following 2 cases, s5
! and s5_1.
subroutine s5(x,y,z,n1,n2)
  use array_of_pointer_test
  type(ta) :: x(:)
  type(tb) :: y(:)
  type(ta), TARGET :: z(:)

  forall (i=1:10)
     ! Convert the rank-1 array to a rank-2 array on assignment
     y(i)%ip(1:n1,1:n2) => z(i)%ip
  end forall
end subroutine s5

! CHECK-LABEL: func @_QPs5(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>> {fir.bindc_name = "y"},
! CHECK-SAME:              %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>> {fir.bindc_name = "z", fir.target},
! CHECK-SAME:              %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "n1"},
! CHECK-SAME:              %[[VAL_4:.*]]: !fir.ref<i32> {fir.bindc_name = "n2"}) {
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_11]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>) {
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:           %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_23]] : index
! CHECK:           %[[VAL_28:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:           %[[VAL_29:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_27]], %[[VAL_28]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, index, !fir.field) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
! CHECK:           %[[VAL_34:.*]] = arith.subi %[[VAL_33]], %[[VAL_30]] : index
! CHECK:           %[[VAL_35:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_29]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
! CHECK:           %[[VAL_37:.*]] = fir.shape_shift %[[VAL_17]], %[[VAL_19]], %[[VAL_20]], %[[VAL_22]] : (i64, i64, i64, i64) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_38:.*]] = fir.rebox %[[VAL_36]](%[[VAL_37]]) : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
! CHECK:           %[[VAL_39:.*]] = fir.array_update %[[VAL_15]], %[[VAL_38]], %[[VAL_34]], %[[VAL_35]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>, !fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>
! CHECK:           fir.result %[[VAL_39]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_40:.*]] to %[[VAL_1]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{ip:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>>
! CHECK:         return
! CHECK:       }

! RHS is an array section. Hits a TODO.
subroutine s5_1(x,y,z,n1,n2)
  use array_of_pointer_test
  type(ta) :: x(:)
  type(tb) :: y(:)
  type(ta), TARGET :: z(:)

  forall (i=1:10)
     ! Slice a rank 1 array and save the slice to the box.
!     x(i)%ip => z(i)%ip(1::n1+1)
  end forall
end subroutine s5_1

subroutine s6(x,y)
  use array_of_pointer_test
  type(tv) :: x(:)
  integer, target :: y(:)

  forall (i=1:10, j=2:20:2)
     ! Two box indirections.
     x(i)%jp(j)%ip%v = y(i)
  end forall
end subroutine s6

! CHECK-LABEL: func @_QPs6(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y", fir.target}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 20 : i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_15]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>) {
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_18]] : (index) -> i32
! CHECK:           fir.store %[[VAL_20]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = fir.do_loop %[[VAL_22:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_14]] unordered iter_args(%[[VAL_23:.*]] = %[[VAL_19]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>) {
! CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_22]] : (index) -> i32
! CHECK:             fir.store %[[VAL_24]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:             %[[VAL_29:.*]] = arith.subi %[[VAL_28]], %[[VAL_25]] : index
! CHECK:             %[[VAL_30:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_29]] : (!fir.array<?xi32>, index) -> i32
! CHECK:             %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
! CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_31]] : index
! CHECK:             %[[VAL_36:.*]] = fir.field_index jp, !fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>
! CHECK:             %[[VAL_37:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
! CHECK:             %[[VAL_40:.*]] = arith.subi %[[VAL_39]], %[[VAL_31]] : index
! CHECK:             %[[VAL_41:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>
! CHECK:             %[[VAL_42:.*]] = fir.field_index v, !fir.type<_QMarray_of_pointer_testTu{v:i32}>
! CHECK:             %[[VAL_43:.*]] = fir.array_access %[[VAL_23]], %[[VAL_35]], %[[VAL_36]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>, index, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>>
! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>>
! CHECK:             %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_44]], %[[VAL_40]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>, index) -> !fir.ref<!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>
! CHECK:             %[[VAL_46:.*]] = fir.coordinate_of %[[VAL_45]], %[[VAL_41]] : (!fir.ref<!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>
! CHECK:             %[[VAL_47:.*]] = fir.load %[[VAL_46]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>>
! CHECK:             %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_47]], %[[VAL_42]] : (!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>, !fir.field) -> !fir.ref<i32>
! CHECK:             fir.store %[[VAL_30]] to %[[VAL_48]] : !fir.ref<i32>
! CHECK:             %[[VAL_49:.*]] = fir.array_amend %[[VAL_23]], %[[VAL_43]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>
! CHECK:             fir.result %[[VAL_49]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_50:.*]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_15]], %[[VAL_51:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{jp:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{ip:!fir.box<!fir.ptr<!fir.type<_QMarray_of_pointer_testTu{v:i32}>>>}>>>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s7(x,y,n)
  use array_of_pointer_test
  type(t) x(:)
  integer, TARGET :: y(:)
  ! Introduce a crossing dependence
  forall (i=1:n)
    x(i)%ip => y(x(n+1-i)%ip)
  end forall
end subroutine s7

! CHECK-LABEL: func @_QPs7(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y", fir.target},
! CHECK-SAME:              %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_17]] : i32
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_18]], %[[VAL_19]] : i32
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_21]], %[[VAL_22]] : i64
! CHECK:           %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_23]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>, i64) -> !fir.ref<!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           %[[VAL_25:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_26:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_25]] : (!fir.ref<!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_28:.*]] = fir.box_addr %[[VAL_27]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_28]] : !fir.ptr<i32>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_31]], %[[VAL_15]] : index
! CHECK:           %[[VAL_33:.*]] = fir.array_access %[[VAL_10]], %[[VAL_32]] : (!fir.array<?xi32>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<i32>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_35:.*]] = fir.embox %[[VAL_34]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
! CHECK:           %[[VAL_40:.*]] = arith.subi %[[VAL_39]], %[[VAL_36]] : index
! CHECK:           %[[VAL_41:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>
! CHECK:           %[[VAL_42:.*]] = fir.array_update %[[VAL_13]], %[[VAL_35]], %[[VAL_40]], %[[VAL_41]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.ptr<i32>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:           fir.result %[[VAL_42]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_43:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s8(x,y,n)
  use array_of_pointer_test
  type(ta) x(:)
  integer, POINTER :: y(:)
  forall (i=1:n)
     x(i)%ip(i:) => y
  end forall
end subroutine s8

! CHECK-LABEL: func @_QPs8(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:              %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "y"},
! CHECK-SAME:              %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_13:.*]] = fir.shift %[[VAL_12]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_9]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) {
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.rebox %[[VAL_10]](%[[VAL_13]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_21]] : index
! CHECK:           %[[VAL_26:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:           %[[VAL_27:.*]] = fir.shift %[[VAL_19]] : (i64) -> !fir.shift<1>
! CHECK:           %[[VAL_28:.*]] = fir.rebox %[[VAL_20]](%[[VAL_27]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_29:.*]] = fir.array_update %[[VAL_16]], %[[VAL_28]], %[[VAL_25]], %[[VAL_26]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:           fir.result %[[VAL_29]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_30:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s8_1(x,y,n1,n2)
  use array_of_pointer_test
  type(ta) x(:)
  integer, POINTER :: y(:)
  forall (i=1:n1)
     x(i)%ip(i:n2+1+i) => y
  end forall
end subroutine s8_1

! CHECK-LABEL: func @_QPs8_1(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>> {fir.bindc_name = "x"},
! CHECK-SAME:                %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "y"},
! CHECK-SAME:                %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n1"},
! CHECK-SAME:                %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "n2"}) {
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_11]], %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_14:.*]] = fir.shift %[[VAL_13]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_9]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_10]]) -> (!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i32
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : i32
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:           %[[VAL_27:.*]] = fir.rebox %[[VAL_11]](%[[VAL_14]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_31]], %[[VAL_28]] : index
! CHECK:           %[[VAL_33:.*]] = fir.field_index ip, !fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:           %[[VAL_34:.*]] = fir.shape_shift %[[VAL_20]], %[[VAL_26]] : (i64, i64) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_35:.*]] = fir.rebox %[[VAL_27]](%[[VAL_34]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_36:.*]] = fir.array_update %[[VAL_17]], %[[VAL_35]], %[[VAL_32]], %[[VAL_33]] : (!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>, index, !fir.field) -> !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:           fir.result %[[VAL_36]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_37:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{ip:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>
! CHECK:         return
! CHECK:       }

subroutine s8_2(x,y,n)
  use array_of_pointer_test
  type(ta) x(:)
  integer, TARGET :: y(:)
  forall (i=1:n)
!     x(i)%ip(i:) => y
  end forall
end subroutine s8_2

subroutine s8_3(x,y,n1,n2)
  use array_of_pointer_test
  type(ta) x(:)
  integer, TARGET :: y(:)
  forall (i=1:n1)
!     x(i)%ip(i:n2+1+i) => y
  end forall
end subroutine s8_3

subroutine s8_4(x,y,n)
  use array_of_pointer_test
  type(ta) x(:)
  integer, ALLOCATABLE, TARGET :: y(:)
  forall (i=1:n)
!     x(i)%ip(i:) => y
  end forall
end subroutine s8_4

subroutine s8_5(x,y,n1,n2)
  use array_of_pointer_test
  type(ta) x(:)
  integer, ALLOCATABLE, TARGET :: y(:)
  forall (i=1:n1)
!     x(i)%ip(i:n2+1+i) => y
  end forall
end subroutine s8_5
