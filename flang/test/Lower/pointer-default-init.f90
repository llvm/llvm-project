! Test that pointer and pointer components are always initialized to a
! clean NULL() status. This is required by f18 runtime to do pointer
! association with a RHS with an undefined association status from a
! Fortran point of view.
! RUN: bbc -emit-fir -I nw %s -o - | FileCheck %s

module test
  type t
    integer :: i
    real, pointer :: x(:)
  end type

  real, pointer :: test_module_pointer(:)
! CHECK-LABEL:   fir.global @_QMtestEtest_module_pointer : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
! CHECK:  %[[VAL_0:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:.*]] = fir.embox %[[VAL_0]](%[[VAL_2]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.has_value %[[VAL_3]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>

  type(t) :: test_module_var
! CHECK-LABEL:   fir.global @_QMtestEtest_module_var : !fir.type<_QMtestTt{i:i32,x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}> {
! CHECK:  %[[VAL_0:.*]] = fir.undefined !fir.type<_QMtestTt{i:i32,x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:  %[[VAL_1:.*]] = fir.undefined i32
! CHECK:  %[[VAL_2:.*]] = fir.field_index i
! CHECK:  %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_1]]
! CHECK:  %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  %[[VAL_8:.*]] = fir.field_index x
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_7]]
! CHECK:  fir.has_value %[[VAL_9]]
end module

subroutine test_local()
  use test, only : t
  type(t) :: x
end subroutine
! CHECK-LABEL:   func.func @_QPtest_local() {
! CHECK:  fir.call @_FortranAInitialize(

subroutine test_saved()
  use test, only : t
  type(t), save :: x
end subroutine
! See check for fir.global internal @_QFtest_savedEx below.

subroutine test_alloc(x)
  use test, only : t
  type(t), allocatable :: x
  allocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_alloc(
! CHECK:  fir.call @_FortranAAllocatableAllocate

subroutine test_intentout(x)
  use test, only : t
  type(t), intent(out):: x
end subroutine
! CHECK-LABEL:   func.func @_QPtest_intentout(
! CHECK-NOT:           fir.call @_FortranAInitialize(
! CHECK:  return

subroutine test_struct_ctor_cst(x)
  use test, only : t
  type(t):: x
  x = t(42)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_struct_ctor_cst(
! CHECK:  fir.call @_FortranAInitialize(

subroutine test_struct_ctor_dyn(x, i)
  use test, only : t
  type(t):: x
  integer :: i
  x = t(i)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_struct_ctor_dyn(
! CHECK:  fir.call @_FortranAInitialize(

subroutine test_local_pointer()
  real, pointer :: x(:)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_local_pointer() {
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "x", uniq_name = "_QFtest_local_pointerEx"}
! CHECK:  %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>

subroutine test_saved_pointer()
  real, pointer, save :: x(:)
end subroutine
! See check for fir.global internal @_QFtest_saved_pointerEx below.

! CHECK-LABEL:   fir.global internal @_QFtest_savedEx : !fir.type<_QMtestTt{i:i32,x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}> {
! CHECK:  %[[VAL_0:.*]] = fir.undefined !fir.type<_QMtestTt{i:i32,x:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:  %[[VAL_1:.*]] = fir.undefined i32
! CHECK:  %[[VAL_2:.*]] = fir.field_index i
! CHECK:  %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_1]]
! CHECK:  %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  %[[VAL_8:.*]] = fir.field_index x
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_7]]
! CHECK:  fir.has_value %[[VAL_9]]

! CHECK-LABEL:   fir.global internal @_QFtest_saved_pointerEx : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
! CHECK:  %[[VAL_0:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:.*]] = fir.embox %[[VAL_0]](%[[VAL_2]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.has_value %[[VAL_3]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
