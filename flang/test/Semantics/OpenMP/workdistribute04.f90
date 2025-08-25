! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! OpenMP Version 6.0
! workdistribute Construct
! Unsuported OpenMP version

subroutine teams_workdistribute()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  !ERROR: WORKDISTRIBUTE construct is not allowed in OpenMP v5.0, try -fopenmp-version=60
  !$omp teams workdistribute
  y = a * x + y
  !$omp end teams workdistribute
end subroutine teams_workdistribute
