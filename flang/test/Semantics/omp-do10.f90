! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The do-loop iteration variable must be of type integer.

program omp_do
  real i, j, k

  !$omp do
  !ERROR: The do-loop iteration variable must be of type integer.
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end program omp_do
