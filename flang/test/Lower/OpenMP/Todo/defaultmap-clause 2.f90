!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=45 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: DEFAULTMAP clause is not implemented yet
subroutine f00
  !$omp target defaultmap(tofrom:scalar)
  !$omp end target
end
