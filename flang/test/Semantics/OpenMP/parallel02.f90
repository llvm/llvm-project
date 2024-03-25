! RUN: not %flang -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s
! OpenMP Version 4.5
! 2.5 parallel construct.
! A program that branches into or out of a parallel region
! is non-conforming.

program omp_parallel
  integer i, j, k

  !CHECK: invalid branch into an OpenMP structured block
  goto 10

  !$omp parallel
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
      !CHECK: In the enclosing PARALLEL directive branched into
      10 stop
    end do
  end do
  !$omp end parallel

end program omp_parallel
