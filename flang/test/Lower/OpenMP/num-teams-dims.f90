! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=61 %s -o - | FileCheck %s

!===============================================================================
! `num_teams` clause with dims modifier (OpenMP 6.1)
!===============================================================================

! CHECK-LABEL: func @_QPteams_numteams_dims2
subroutine teams_numteams_dims2()
  ! CHECK: omp.teams
  ! CHECK-SAME: num_teams( to %{{.*}}, %{{.*}} : i32, i32)
  !$omp teams num_teams(dims(2): 10, 4)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_numteams_dims2

! CHECK-LABEL: func @_QPteams_numteams_dims3
subroutine teams_numteams_dims3()
  ! CHECK: omp.teams
  ! CHECK-SAME: num_teams( to %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32)
  !$omp teams num_teams(dims(3): 8, 4, 2)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_numteams_dims3

! CHECK-LABEL: func @_QPteams_numteams_dims_var
subroutine teams_numteams_dims_var(a, b, c)
  integer, intent(in) :: a, b, c
  ! CHECK: omp.teams
  ! CHECK-SAME: num_teams( to %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32)
  !$omp teams num_teams(dims(3): a, b, c)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_numteams_dims_var

!===============================================================================
! `num_teams` clause with lower bound (legacy, without dims)
!===============================================================================

! CHECK-LABEL: func @_QPteams_numteams_lower_upper
subroutine teams_numteams_lower_upper(lower, upper)
  integer, intent(in) :: lower, upper
  ! CHECK: omp.teams
  ! CHECK-SAME: num_teams(%{{.*}} : i32 to %{{.*}} : i32)
  !$omp teams num_teams(lower: upper)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_numteams_lower_upper
