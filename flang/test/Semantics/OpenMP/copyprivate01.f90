! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.4.2 copyprivate Clause
! A list item that appears in a copyprivate clause may not appear in a
! private or firstprivate clause on the single construct.

program omp_copyprivate
  integer :: a(10), b(10), k

  k = 10
  a = 10
  b = a * 10

  !$omp parallel
  !$omp single private(k)
  a = a + k
  !ERROR: COPYPRIVATE variable 'k' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
  !$omp end single copyprivate(k)
  !$omp single firstprivate(k)
  b = a - k
  !ERROR: COPYPRIVATE variable 'k' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
  !$omp end single copyprivate(k)
  !$omp end parallel

  print *, a, b

end program omp_copyprivate
