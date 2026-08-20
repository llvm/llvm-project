!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPtask_threadset_pool
subroutine task_threadset_pool()
  !CHECK: omp.task threadset(omp_pool) {
  !$omp task threadset(omp_pool)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_threadset_pool

!CHECK-LABEL: func @_QPtask_threadset_team
subroutine task_threadset_team()
  !CHECK: omp.task threadset(omp_team) {
  !$omp task threadset(omp_team)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_threadset_team

!CHECK-LABEL: func @_QPtaskloop_threadset_pool
subroutine taskloop_threadset_pool()
  integer :: i
  !CHECK: omp.taskloop.context threadset(omp_pool)
  !$omp taskloop threadset(omp_pool)
  do i = 1, 10
  end do
  !$omp end taskloop
end subroutine taskloop_threadset_pool

!CHECK-LABEL: func @_QPtaskloop_threadset_team
subroutine taskloop_threadset_team()
  integer :: i
  !CHECK: omp.taskloop.context threadset(omp_team)
  !$omp taskloop threadset(omp_team)
  do i = 1, 10
  end do
  !$omp end taskloop
end subroutine taskloop_threadset_team
