! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5

integer :: x
!ERROR: Variable 'x' may not appear on both MAP and PRIVATE clauses on a TARGET construct
!$omp target map(x) private(x)
x = x + 1
!$omp end target

end
