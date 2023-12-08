! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP Block construct clause
subroutine sb
  integer, save :: a
  !$omp threadprivate(a)
  !$omp parallel
  !$omp single
  a = 3
  !$omp end single copyprivate(a)
  !$omp end parallel
end subroutine
