! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

!===============================================================================
! `untied` clause
!===============================================================================

! CHECK: not yet implemented: UNTIED clause is not implemented yet
subroutine omp_task_untied()
  !$omp task untied
  call foo()
  !$omp end task
end subroutine omp_task_untied
