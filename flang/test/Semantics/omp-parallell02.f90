! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.5 parallel construct.
! A program that branches into or out of a parallel region
! is non-conforming.

program omp_parallel
  integer i, j, k

  !ERROR: invalid entry to OpenMP structured block
  goto 10

  !$omp parallel
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
      10 stop
    end do
  end do
  !$omp end parallel

end program omp_parallel
