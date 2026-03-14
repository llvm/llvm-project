! Test lowering of whole allocatable and pointer components to HLFIR
! RUN: bbc -emit-hlfir -o - %s -I nw | FileCheck %s

module def_test_types
  type t1
    real, pointer :: p(:)
  end type
  type t2
    real, allocatable :: a(:)
  end type
  type t3
    real, pointer, contiguous :: p_contiguous(:)
  end type
  type t4
    character(:), pointer :: char_p(:)
  end type
  type t5
    character(10), allocatable :: char_a(:)
  end type
  interface
    subroutine takes_pointer(y)
      real, pointer :: y(:)
    end subroutine
    subroutine takes_contiguous_pointer(y)
      real, pointer, contiguous :: y(:)
    end subroutine
    subroutine takes_allocatable(y)
      real, allocatable :: y(:)
    end subroutine
    subroutine takes_char_pointer(y)
      character(:), pointer :: y(:)
    end subroutine
    subroutine takes_char_alloc_cst_len(y)
      character(10), allocatable :: y(:)
    end subroutine
    subroutine takes_array(y)
      real :: y(*)
    end subroutine
    subroutine takes_char_array(y)
      character(*) :: y(*)
    end subroutine

  end interface
end module

subroutine passing_pointer(x)
  use  def_test_types
  implicit none
  type(t1) :: x
  call takes_pointer(x%p)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMdef_test_typesTt1{p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  fir.call @_QPtakes_pointer(%[[VAL_2]]) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()

subroutine passing_allocatable(x)
  use  def_test_types
  implicit none
  type(t2) :: x
  call takes_allocatable(x%a)
  call takes_array(x%a)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_allocatable(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"a"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMdef_test_typesTt2{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  fir.call @_QPtakes_allocatable(%[[VAL_2]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
! CHECK:  %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]]#0{"a"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMdef_test_typesTt2{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_array(%[[VAL_6]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()

subroutine passing_contiguous_pointer(x)
  use  def_test_types
  type(t3) :: x
  call takes_contiguous_pointer(x%p_contiguous)
  call takes_array(x%p_contiguous)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_contiguous_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"p_contiguous"}   {fortran_attrs = #fir.var_attrs<contiguous, pointer>} : (!fir.ref<!fir.type<_QMdef_test_typesTt3{p_contiguous:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  fir.call @_QPtakes_contiguous_pointer(%[[VAL_2]]) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()
! CHECK:  %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]]#0{"p_contiguous"}   {fortran_attrs = #fir.var_attrs<contiguous, pointer>} : (!fir.ref<!fir.type<_QMdef_test_typesTt3{p_contiguous:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_array(%[[VAL_6]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()

subroutine passing_char_pointer(x)
  use  def_test_types
  implicit none
  type(t4) :: x
  call takes_char_pointer(x%char_p)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_char_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"char_p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMdef_test_typesTt4{char_p:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  fir.call @_QPtakes_char_pointer(%[[VAL_2]]) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()

subroutine passing_char_alloc_cst_len(x)
  use  def_test_types
  implicit none
  type(t5) :: x
  call takes_char_alloc_cst_len(x%char_a)
  call takes_char_array(x%char_a)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_char_alloc_cst_len(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]]#0{"char_a"}   typeparams %[[VAL_2]] {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMdef_test_typesTt5{char_a:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>}>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  fir.call @_QPtakes_char_alloc_cst_len(%[[VAL_3]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) -> ()
! CHECK:  %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_1]]#0{"char_a"}   typeparams %[[VAL_4]] {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMdef_test_typesTt5{char_a:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>}>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,10>>
! CHECK:  %[[VAL_9:.*]] = fir.emboxchar %[[VAL_8]], %[[VAL_4]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPtakes_char_array(%[[VAL_9]]) {{.*}}: (!fir.boxchar<1>) -> ()
