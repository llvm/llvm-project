! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! collapse(n) where n > num of loops

program omp_do
  integer i, j, k

  !ERROR: Not enough do loops for collapsed !$OMP DO
  !$omp do collapse(2)
  do i = 1, 10
    print *, "hello"
  end do
  !$omp end do

end program omp_do
