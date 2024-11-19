!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: PRESENT modifier is not supported yet
subroutine f00(x)
  integer :: x
  !$omp target update from(present: x)
end
