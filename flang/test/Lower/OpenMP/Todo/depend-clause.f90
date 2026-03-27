!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!CHECK: Support for iterator modifiers is not implemented yet
subroutine f00(x)
  integer :: x(10)
  !$omp task depend(iterator(i = 1:10), in: x(i))
  x = 0
  !$omp end task
end
