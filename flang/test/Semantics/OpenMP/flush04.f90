! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! Regression test to ensure that the name /c/ in the flush argument list is
! resolved to the common block symbol.

  common /c/ x
  real :: x
!ERROR: FLUSH argument must be a variable list item
  !$omp flush(/c/)
end

