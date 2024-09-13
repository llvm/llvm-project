! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! OpenMP Version 4.5
! 2.8.1 simd Construct
! Semantic error for correct test case

program omp_simd
  integer i, j, k
  integer, allocatable :: a(:), b(:)

  allocate(a(10))
  allocate(b(10))

  !ERROR: List item 'a' present at multiple NONTEMPORAL clauses
  !$omp simd nontemporal(a, a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple NONTEMPORAL clauses
  !ERROR: List item 'b' present at multiple NONTEMPORAL clauses
  !$omp simd nontemporal(a,a) nontemporal(b) nontemporal(b)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple NONTEMPORAL clauses
  !$omp simd nontemporal(a) nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd

  !$omp simd nontemporal(a) nontemporal(b)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple NONTEMPORAL clauses
  !$omp simd nontemporal(a) private(a) nontemporal(a)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  !ERROR: List item 'a' present at multiple NONTEMPORAL clauses
  !ERROR: List item 'b' present at multiple NONTEMPORAL clauses
  !$omp simd nontemporal(a,a,b,b)
  do i = 1, 10
    a(i) = i
    b(i) = i
  end do
  !$omp end simd

  print *, a

end program omp_simd
