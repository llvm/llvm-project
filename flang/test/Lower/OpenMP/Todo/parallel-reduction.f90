! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP Block construct clauses
subroutine reduction_parallel
  integer :: x
  !$omp parallel reduction(+:x)
  x = x + i
  !$omp end parallel
  print *, x
end subroutine
