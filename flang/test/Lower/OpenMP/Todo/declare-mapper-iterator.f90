!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

!CHECK: Support for iterator modifiers is not implemented yet
subroutine f(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m: s :: v) map(mapper(m), iterator(i = 1:10): v%a(i))
end
