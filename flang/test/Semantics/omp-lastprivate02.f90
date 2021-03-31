! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.15.3.5 lastprivate Clause
! A list item that is private within a parallel region, or that appears in
! reduction clause of a parallel construct, must not appear in a
! lastprivate clause on a worksharing construct if any of the corresponding
! worksharing regions ever binds to any of the corresponding parallel regions.

program omp_lastprivate
  integer :: a(10), b(10), c(10)

  a = 10
  b = 20

  !$omp parallel reduction(+:a)
  !ERROR: LASTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp sections lastprivate(a, b)
  !$omp section
  c = a + b
  !$omp end sections
  !$omp end parallel

  !$omp parallel private(a,b)
  !ERROR: LASTPRIVATE variable 'a' is PRIVATE in outer context
  !ERROR: LASTPRIVATE variable 'b' is PRIVATE in outer context
  !$omp do lastprivate(a,b)
  do i = 1, 10
    c(i) = a(i) + b(i) + i
  end do
  !$omp end do
  !$omp end parallel

  print *, c

end program omp_lastprivate
