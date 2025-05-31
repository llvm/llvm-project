! RUN: %python %S/test_folding.py %s %flang_fc1
! Fold NORM2 reduction of array with non-default lower bound
module m
  real, parameter :: a(2:3) = 0.0
  logical, parameter :: test1 = norm2(a) == 0.
end
