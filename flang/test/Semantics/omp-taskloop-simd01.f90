! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.9.3 taskloop simd Construct
! No reduction clause may be specified for !$omp taskloop simd.

program omp_taskloop_simd
  integer i , j , k

  !ERROR: Unexpected clause specified for !$OMP taskloop simd
  !$omp taskloop simd reduction(+:k)
  do i=1,10000
    do j=1,i
      call loop_body(i, j)
      k = k + 1
    end do
  end do
  !$omp end taskloop simd

end program omp_taskloop_simd
