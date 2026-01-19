! RUN: not %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 %s -o - 2>&1 | FileCheck %s

! CHECK: error: teams has multiple workdistribute ops.
! CHECK-LABEL: func @_QPteams_workdistribute_1
subroutine teams_workdistribute_1()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  !$omp teams

  !$omp workdistribute
  y = a * x + y
  !$omp end workdistribute

  !$omp workdistribute
  y = a * y + x
  !$omp end workdistribute
  !$omp end teams
end subroutine teams_workdistribute_1
