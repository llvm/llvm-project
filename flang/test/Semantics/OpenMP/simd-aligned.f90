! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! OpenMP Version 4.5
! 2.8.1 simd Construct
! Semantic error for correct test case

program omp_simd
  integer i, j, k, c, d(100)
  integer, allocatable :: a(:), b(:)
  common /cmn/ c

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

  !ERROR: 'c' is a common block name and can not appear in an ALIGNED clause
  !$omp simd aligned(c)
  do i = 1, 10
    c = 5
  end do
  !$omp end simd

  !ERROR: 'd' in ALIGNED clause must be of type C_PTR, POINTER or ALLOCATABLE
  !WARNING: Alignment is not a power of 2, Aligned clause will be ignored [-Wopen-mp-usage]
  !$omp simd aligned(d:100)
  do i = 1, 100
    d(i) = i
  end do

  !WARNING: Alignment is not a power of 2, Aligned clause will be ignored [-Wopen-mp-usage]
  !$omp simd aligned(b:65)
  do i = 1, 100
    b(i) = i
  end do

end program omp_simd
