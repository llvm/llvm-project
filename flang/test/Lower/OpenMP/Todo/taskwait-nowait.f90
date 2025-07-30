! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s -fopenmp-version=51 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s -fopenmp-version=51 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause NOWAIT in TASKWAIT construct
subroutine omp_tw_nowait
  !$omp taskwait nowait
end subroutine omp_tw_nowait

