! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

!===============================================================================
! `mergeable` clause
!===============================================================================

! CHECK: not yet implemented: Unhandled clause IN_REDUCTION in TASK construct
subroutine omp_task_in_reduction()
  integer i
  i = 0
  !$omp task in_reduction(+:i)
  i = i + 1
  !$omp end task
end subroutine omp_task_in_reduction
