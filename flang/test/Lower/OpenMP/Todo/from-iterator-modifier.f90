!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: Iterator modifier is not supported yet
subroutine f00(x)
  integer :: x(10)
  !$omp target update from(iterator(i = 1:2): x(i))
end
