! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause REDUCTION in TEAMS construct
subroutine reduction_teams()
  integer :: i
  i = 0

  !$omp teams reduction(+:i)
  i = i + 1
  !$omp end teams
end subroutine reduction_teams
