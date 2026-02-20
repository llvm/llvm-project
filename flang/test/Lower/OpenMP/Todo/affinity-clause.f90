!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!CHECK: Support for iterator modifiers is not implemented yet
subroutine f00(x)
  integer :: x(10)
!$omp task affinity(iterator(i = 1:10) : x(i))
  x = x + 1
!$omp end task
end
