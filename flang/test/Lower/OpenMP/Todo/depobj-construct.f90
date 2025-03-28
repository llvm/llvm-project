!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: OpenMPDepobjConstruct
subroutine f00()
  integer :: obj
  integer :: x
  !$omp depobj(obj) depend(in: x)
end
