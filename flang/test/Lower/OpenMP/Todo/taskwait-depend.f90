! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause DEPEND in TASKWAIT construct
subroutine omp_tw_depend
  integer :: res
  !$omp taskwait depend(out: res)
  res = res + 1
end subroutine omp_tw_depend

