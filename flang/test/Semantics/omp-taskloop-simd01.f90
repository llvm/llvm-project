! RUN: %python %S/test_errors.py %s %flang -fopenmp

! OpenMP Version 5.0
! 2.10.3 taskloop simd Construct

program omp_taskloop_simd
  integer i , j , k

  !$omp taskloop simd reduction(+:k)
  do i=1,10000
    do j=1,i
      k = k + 1
    end do
  end do
  !$omp end taskloop simd

end program omp_taskloop_simd
