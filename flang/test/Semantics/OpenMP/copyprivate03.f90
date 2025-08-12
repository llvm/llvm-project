! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.4.2 copyprivate Clause
! All list items that appear in the copyprivate clause must be either
! threadprivate or private in the enclosing context.

program omp_copyprivate
  integer :: a(10), b(10)
  real, dimension(:), allocatable :: c
  real, dimension(:), pointer :: d
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

  !The use of FIRSTPRIVATE with COPYPRIVATE is allowed
  !$omp parallel firstprivate(a)
  !$omp single
  a = a + k
  !$omp end single copyprivate(a)
  !$omp end parallel

  print *, a, b

  !$omp task
    !$omp parallel private(c, d)
      allocate(c(5))
      allocate(d(10))
      !$omp single
        c = 22
        d = 33
      !Check that 'c' and 'd' inherit PRIVATE DSA from the enclosing PARALLEL
      !and no error occurs.
      !$omp end single copyprivate(c, d)
    !$omp end parallel
  !$omp end task
end program omp_copyprivate
