! RUN: %S/test_errors.sh %s %t %flang -fopenmp

! OpenMP Version 4.5
! 2.8.1 simd Construct
! Semantic error for correct test case

program omp_simd
  integer i, j, k
  integer, allocatable :: a(:), b(:)

  allocate(a(10))
  allocate(b(10))

  !ERROR: List item 'a' present at multiple ALIGNED clauses
  !$omp simd aligned(a, a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple ALIGNED clauses
  !ERROR: List item 'b' present at multiple ALIGNED clauses
  !$omp simd aligned(a,a) aligned(b) aligned(b)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple ALIGNED clauses
  !$omp simd aligned(a) aligned(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd

  !$omp simd aligned(a) aligned(b)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple ALIGNED clauses
  !$omp simd aligned(a) private(a) aligned(a)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  print *, a

end program omp_simd
