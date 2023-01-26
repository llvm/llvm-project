! Test lowering of allocatable and pointer sub-part reference to HLFIR
! As opposed to whole reference, a pointer/allocatable dereference must
! be inserted and addressed in a following hlfir.designate to address
! the sub-part.

! RUN: bbc -emit-fir -hlfir -o - %s -I nw | FileCheck %s

module m
  type t1
    real :: x
  end type
  type t2
    type(t1), pointer :: p
  end type
  type t3
    character(:), allocatable :: a(:)
  end type
end module

subroutine test_pointer_component_followed_by_component_ref(x)
  use m
  type(t2) :: x
  call takes_real(x%p%x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_pointer_component_followed_by_component_ref(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}} {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt2{p:!fir.box<!fir.ptr<!fir.type<_QMmTt1{x:f32}>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt1{x:f32}>>>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt1{x:f32}>>>>
! CHECK:  %[[VAL_4:.*]] = hlfir.designate %[[VAL_3]]{"x"}   : (!fir.box<!fir.ptr<!fir.type<_QMmTt1{x:f32}>>>) -> !fir.ref<f32>

subroutine test_symbol_followed_by_ref(x)
  character(:), allocatable :: x(:)
  call test_char(x(10))
end subroutine
! CHECK-LABEL: func.func @_QPtest_symbol_followed_by_ref(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:  %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_2]] (%[[VAL_4]])  typeparams %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index, index) -> !fir.boxchar<1>

subroutine test_component_followed_by_ref(x)
  use m
  type(t3) :: x
  call test_char(x%a(10))
end subroutine
! CHECK-LABEL: func.func @_QPtest_component_followed_by_ref(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}} {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"a"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt3{a:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_elesize %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:  %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_6:.*]] = hlfir.designate %[[VAL_3]] (%[[VAL_5]])  typeparams %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index, index) -> !fir.boxchar<1>
