! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: iterator modifier with derived type member map

subroutine map_iterator_array_derived_member
  type :: t
    integer :: b
  end type
  type(t) :: x(10)
  integer :: i

  !$omp target data map(iterator(i = 1:10), tofrom: x(i)%b)
  !$omp end target data
end subroutine
