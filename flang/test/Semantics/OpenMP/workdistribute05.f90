! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60
! OpenMP Version 6.0
! workdistribute Construct
! Make sure that standalone workdistribute doesn't trigger errors when used in
! supported contexts.

subroutine teams_workdistribute()
  integer :: a
  !$omp teams
  !$omp workdistribute
  a = 1
  !$omp end workdistribute
  !$omp end teams
end subroutine teams_workdistribute

subroutine target_teams_workdistribute()
  integer :: a
  !$omp target teams
  !$omp workdistribute
  a = 1
  !$omp end workdistribute
  !$omp end target teams
end subroutine target_teams_workdistribute
