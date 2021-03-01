! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.15.4.2 copyprivate Clause
! All list items that appear in the copyprivate clause must be either
! threadprivate or private in the enclosing context.

program omp_copyprivate
  integer :: a(10), b(10)
  integer, save :: k

  !$omp threadprivate(k)

  k = 10
  a = 10
  b = a + 10

  !$omp parallel
  !$omp single
  a = a + k
  !$omp end single copyprivate(k)
  !$omp single
  b = b - a
  !ERROR: COPYPRIVATE variable 'b' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(b)
  !$omp end parallel

  !$omp parallel sections private(a)
  !$omp section
  !$omp parallel
  !$omp single
  a = a * b + k
  !ERROR: COPYPRIVATE variable 'a' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(a)
  !$omp end parallel
  !$omp end parallel sections

  print *, a, b

end program omp_copyprivate
