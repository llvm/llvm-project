!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: ATTACH modifier is not implemented yet
subroutine f00(x)
  integer, pointer :: x
  !$omp target map(attach(always), tofrom: x)
  !$omp end target
end
