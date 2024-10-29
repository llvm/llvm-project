! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: Map type modifiers (other than 'ALWAYS') are not implemented yet
subroutine f02()
  integer :: x
  !$omp target map(present: x)
  x = x + 1
  !$omp end target
end

