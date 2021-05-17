! RUN: not %flang -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s
! OpenMP Version 4.5
! 2.8.1 simd Construct
! A program that branches into or out of a simd region is non-conforming.

program omp_simd
  integer i, j

  !$omp simd
  do i = 1, 10
    do j = 1, 10
      print *, "omp simd"
      !CHECK: invalid branch leaving an OpenMP structured block
      goto 10
    end do
  end do
  !$omp end simd

  !CHECK: Outside the enclosing SIMD directive
  10 stop

end program omp_simd
