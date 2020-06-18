! RUN: %S/test_errors.sh %s %t %f18 -fopenmp

! 2.15.3 Data-Sharing Attribute Clauses
! A list item that specifies a given variable may not appear in more than
! one clause on the same directive, except that a variable may be specified
! in both firstprivate and lastprivate clauses.

  common /c/ a, b
  integer a(3), b

  A = 1
  B = 2
  !ERROR: 'c' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel shared(/c/,c) private(/c/)
  a(1:2) = 3
  B = 4
  !$omp end parallel
  print *, a, b, c
end
