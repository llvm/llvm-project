! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!===============================================================================
! `private` clause on `target teams`
!===============================================================================

! CHECK: not yet implemented: TARGET TEAMS PRIVATE is not implemented yet
subroutine target_teams_private()
integer, dimension(3) :: i
!$omp target teams private(i)
!$omp end target teams
end subroutine
