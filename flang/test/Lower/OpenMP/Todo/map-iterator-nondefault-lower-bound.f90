!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: Iterator modifier on a derived-type member with a non-default lower bound is not implemented yet
subroutine f(arg)
  type :: s
    integer :: a(-2:2)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m: s :: v) map(mapper(m), iterator(i = -2:2): v%a(i))
end
