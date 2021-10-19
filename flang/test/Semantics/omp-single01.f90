! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.3 single Construct
! Symbol present on multiple clauses

program omp_single
  integer i
  i = 10

  !$omp single private(i)
  print *, "omp single", i
  !ERROR: COPYPRIVATE variable 'i' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
  !$omp end single copyprivate(i)

end program omp_single
