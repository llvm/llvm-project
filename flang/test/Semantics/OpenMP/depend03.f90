! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.13.9 Depend Clause
! Coarrays are not supported in depend clause

program omp_depend_coarray
  integer :: a(3)[*], b(3) , k

  a(:) = this_image()
  b(:) = a(:)[1]
  k = 10

  !$omp parallel
  !$omp single
  !ERROR: Coarrays are not supported in DEPEND clause
  !$omp task shared(b) depend(out: a(:)[1])
  b = a + k
  !$omp end task
  !$omp end single
  !$omp end parallel

  print *, a, b

end program omp_depend_coarray
