! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of DIM()
module m
  logical, parameter :: test_i1 = dim(0, 0) == 0
  logical, parameter :: test_i2 = dim(1, 2) == 0
  logical, parameter :: test_i3 = dim(2, 1) == 1
  logical, parameter :: test_i4 = dim(2, -1) == 3
  logical, parameter :: test_i5 = dim(-1, 2) == 0
  logical, parameter :: test_a1 = dim(0., 0.) == 0.
  logical, parameter :: test_a2 = dim(1., 2.) == 0.
  logical, parameter :: test_a3 = dim(2., 1.) == 1.
  logical, parameter :: test_a4 = dim(2., -1.) == 3.
  logical, parameter :: test_a5 = dim(-1., 2.) == 0.
  !WARN: warning: invalid argument on division
  real, parameter :: nan = 0./0.
  logical, parameter :: test_a6 = dim(nan, 1.) /= dim(nan, 1.)
end module
