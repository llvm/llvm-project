!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: DYN_GROUPPRIVATE clause is not implemented yet
subroutine f00(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(n)
  !$omp end target
end

