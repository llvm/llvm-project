! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtarget_teams_workdistribute
subroutine target_teams_workdistribute()
  integer :: aa(10), bb(10)
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.workdistribute
  !$omp target teams workdistribute
  aa = bb
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end target teams workdistribute
end subroutine target_teams_workdistribute

! CHECK-LABEL: func @_QPteams_workdistribute
subroutine teams_workdistribute()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  ! CHECK: omp.teams
  ! CHECK: omp.workdistribute
  !$omp teams workdistribute
  y = a * x + y
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end teams workdistribute
end subroutine teams_workdistribute
