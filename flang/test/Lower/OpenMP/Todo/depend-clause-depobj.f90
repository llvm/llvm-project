!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: DEPOBJ dependence-type

subroutine f00(x)
  use iso_c_binding, only: c_intptr_t
  integer(c_intptr_t) :: x
  !$omp task depend(depobj: x)
  !$omp end task
end
