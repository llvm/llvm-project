! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! chunk_size must be a loop invariant integer expression
! with a positive value.

program omp_do
  integer i, j, k
  integer :: a(10), b(10)
  a = 10
  j = 0

  !ERROR: INTEGER expression of SCHEDULE clause chunk_size must be positive
  !$omp do schedule(static, -1)
  do i = 1, 10
    j = j + 1
    b(i) = a(i) * 2.0
  end do
  !$omp end do

  print *, j
  print *, b

end program omp_do
