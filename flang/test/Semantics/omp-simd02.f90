! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell

! OpenMP Version 4.5
! 2.8.1 simd Construct
! Semantic error for correct test case

program omp_simd
  integer i, j, k
  integer, allocatable :: a(:)

  allocate(a(10))

  !$omp simd aligned(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd

  print *, a

end program omp_simd
