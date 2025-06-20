! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!===============================================================================
! `private` clause on `target parallel`
!===============================================================================

! CHECK: not yet implemented: TARGET PARALLEL PRIVATE is not implemented yet
subroutine target_teams_private()
integer, dimension(3) :: i
!$omp target parallel private(i)
!$omp end target parallel
end subroutine
