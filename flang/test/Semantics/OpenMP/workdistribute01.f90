! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60
! OpenMP Version 6.0
! workdistribute Construct
! Invalid do construct inside !$omp workdistribute

subroutine workdistribute()
  integer n, i
  !ERROR: A WORKDISTRIBUTE region must be nested inside TEAMS region only.
  !ERROR: The structured block in a WORKDISTRIBUTE construct may consist of only SCALAR or ARRAY assignments
  !$omp workdistribute
  do i = 1, n
    print *, "omp workdistribute"
  end do
  !$omp end workdistribute

end subroutine workdistribute
