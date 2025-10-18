! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtarget_teams_workdistribute
subroutine target_teams_workdistribute(a, x, y, rows, cols)
  use iso_fortran_env
  implicit none

  integer, intent(in) :: rows, cols
  real(kind=real32) :: a
  real(kind=real32), dimension(rows, cols) :: x, y

  ! CHECK: omp.target_data
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest
  ! CHECK: fir.do_loop

  !$omp target teams workdistribute
  y = a * x + y
  !$omp end target teams workdistribute
end subroutine target_teams_workdistribute

! CHECK-LABEL: func @_QPteams_workdistribute
subroutine teams_workdistribute(a, x, y, rows, cols)
  use iso_fortran_env
  implicit none

  integer, intent(in) :: rows, cols
  real(kind=real32) :: a
  real(kind=real32), dimension(rows, cols) :: x, y

  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest
  ! CHECK: fir.do_loop

  !$omp teams workdistribute
  y = a * x + y
  !$omp end teams workdistribute
end subroutine teams_workdistribute
