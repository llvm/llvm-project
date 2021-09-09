! RUN: %python %S/test_errors.py %s %flang -fopenmp

! OpenMP Version 4.5
! 2.8.3 Loop simd Construct
! Semantic error for correct test case.

program omp_loop_simd
  integer i, j, k, l
  k = 0;
  l = 0

  !$omp parallel do simd linear(l)
  do i = 1, 10
    do j = 1, 10
      print *, "omp loop simd"
      k = k + 1
      l = l + 1
    end do
  end do

  print *, k, l

end program omp_loop_simd
