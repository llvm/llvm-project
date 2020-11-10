! RUN: %S/test_errors.sh %s %t %f18 -fopenmp

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! Semantic error for correct test case

program omp_do
  integer i, j, k
  integer :: a(10), b(10)
  a = 10
  j = 0

  !$omp parallel
    !$omp do linear(j:1)
    do i = 1, 10
      j = j + 1
      b(i) = a(i) * 2.0
    end do
    !$omp end do
  !$omp end parallel

  print *, j
  print *, b

end program omp_do
