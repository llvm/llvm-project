! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.13.9 Depend Clause
! List items used in depend clauses cannot be zero-length array sections.

program omp_depend
  integer :: a(10) , b(10,10)
  a = 10
  b = 20

  !$omp parallel
  !$omp single

  !ERROR: 'a' in DEPEND clause is a zero size array section
  !ERROR: 'b' in DEPEND clause is a zero size array section
  !$omp task shared(a,b) depend(out: a(2:1), b(3:1, 1:-1))
  a(2:1) = b(2, 2)
  !$omp end task

  !ERROR: 'a' in DEPEND clause must have a positive stride
  !ERROR: 'b' in DEPEND clause must have a positive stride
  !ERROR: 'b' in DEPEND clause is a zero size array section
  !$omp task shared(x) depend(in: a(10:5:-1)) depend(in: b(5:10:-1))
  print *, a(5:10), b
  !$omp end task

  !ERROR: 'a' in DEPEND clause is a zero size array section
  !$omp task shared(x) depend(in: a(1:5:10))
  print *, a(5:10), b
  !$omp end task

  !$omp end single
  !$omp end parallel

end program omp_depend
