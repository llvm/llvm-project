! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Common Block in privatization clause
subroutine firstprivate_common
  common /c/ x, y
  real x, y
  !$omp parallel firstprivate(/c/)
  !$omp end parallel
end subroutine
