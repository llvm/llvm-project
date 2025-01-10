! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause TASK_REDUCTION in TASKGROUP construct
subroutine omp_taskgroup_task_reduction
  integer :: res
  !$omp taskgroup task_reduction(+:res)
  res = res + 1
  !$omp end taskgroup
end subroutine omp_taskgroup_task_reduction
