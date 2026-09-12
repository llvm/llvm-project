! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! A flat derived type gets no implicit default mapper, even when the captured
! variable is allocatable.

program p
  type t
    integer :: x
    real(8) :: v(4)
  end type

  type(t), allocatable :: al(:)
  allocate(al(100))

  !$omp target
    al(1)%x = al(1)%x + 1
  !$omp end target
end program

! CHECK-NOT: omp.declare_mapper

! CHECK-LABEL: func.func @_QQmain
! CHECK-NOT: mapper(
