! RUN: %S/test_errors.sh %s %t %f18 -fopenmp

! OpenMP Version 4.5
! 2.5 parallel construct.
! A program that branches into or out of a parallel region
! is non-conforming.

program omp_parallel
  integer i, j, k

  !$omp parallel
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
      !ERROR: Control flow escapes from PARALLEL
      goto 10
    end do
  end do
  !$omp end parallel

  10 stop

end program omp_parallel
