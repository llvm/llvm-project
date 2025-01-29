! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction modifiers are not supported
subroutine reduction_task()
  integer :: i
  i = 0

  !$omp parallel reduction(task, +:i)
  i = i + 1
  !$omp end parallel 
end subroutine reduction_task
