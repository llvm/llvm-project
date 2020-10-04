! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.8.1 simd Construct
! A program that branches into or out of a simd region is non-conforming.

program omp_simd
  integer i, j

  !$omp simd
  do i = 1, 10
    do j = 1, 10
      print *, "omp simd"
      !ERROR: invalid branch to/from OpenMP structured block
      goto 10
    end do
  end do
  !$omp end simd

  10 stop

end program omp_simd
