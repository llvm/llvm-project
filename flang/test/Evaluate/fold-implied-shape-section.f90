! RUN: %python %S/test_folding.py %s %flang_fc1
! Omitted-bound sections of implied-shape named constants with nondefault
! lower bounds are constant expressions: LBOUND of the whole array is the
! declared lower bound (1 for an empty dimension, F'2023 16.9.119 p5), and
! the bounds of a section are 1-based.
module m
  integer, parameter :: p0(0:*) = [1, 2, 3]
  integer, parameter :: p2(2:*, 0:*) = reshape([1, 2, 3, 4, 5, 6], [2, 3])
  integer, parameter :: e(5:*) = [integer::]
  logical, parameter :: test_p0_whole_lb = lbound(p0, 1) == 0
  logical, parameter :: test_p0_size = size(p0(:)) == 3
  logical, parameter :: test_p0_lb = lbound(p0(:), 1) == 1
  logical, parameter :: test_p0_ub = ubound(p0(:), 1) == 3
  logical, parameter :: test_p0_vals = all(p0(:) == [1, 2, 3])
  logical, parameter :: test_p2_whole_lb = all(lbound(p2) == [2, 0])
  logical, parameter :: test_p2_size = size(p2(:, :)) == 6
  logical, parameter :: test_p2_shape = all(shape(p2(:, :)) == [2, 3])
  logical, parameter :: test_p2_ub = ubound(p2(:, 1), 1) == 2
  logical, parameter :: test_e_lb = lbound(e, 1) == 1
  logical, parameter :: test_e_sec_lb = lbound(e(:), 1) == 1
  logical, parameter :: test_e_size = size(e(:)) == 0
end module
