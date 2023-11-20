! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

!===============================================================================
! `mergeable` clause
!===============================================================================

! CHECK: not yet implemented: OpenMP Block construct clause
subroutine omp_task_mergeable()
  !$omp task mergeable
  call foo()
  !$omp end task
end subroutine omp_task_mergeable
