! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!===============================================================================
! `mergeable` clause
!===============================================================================

! CHECK: not yet implemented: Unhandled clause IN_REDUCTION in TARGET construct
subroutine omp_target_inreduction()
  integer i
  i = 0
  !$omp target in_reduction(+:i)
  i = i + 1
  !$omp end target
end subroutine omp_target_inreduction
