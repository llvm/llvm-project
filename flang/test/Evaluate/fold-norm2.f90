! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of NORM2(), F'2023 16.9.153
module m
  ! Examples from the standard
  logical, parameter :: test_ex1 = norm2([3.,4.]) == 5.
  real, parameter :: ex2(2,2) = reshape([1.,3.,2.,4.],[2,2])
  real, parameter :: ex2_norm2_1(2) = norm2(ex2, dim=1)
  real, parameter :: ex2_1(2) = [3.162277698516845703125,4.472136020660400390625]
  logical, parameter :: test_ex2_1 = all(ex2_norm2_1 == ex2_1)
  real, parameter :: ex2_norm2_2(2) = norm2(ex2, dim=2)
  real, parameter :: ex2_2(2) = [2.2360680103302001953125,5.]
  logical, parameter :: test_ex2_2 = all(ex2_norm2_2 == ex2_2)
  !  0  3  6  9
  !  1  4  7 10
  !  2  5  8 11
  integer, parameter :: dp = kind(0.d0)
  real(dp), parameter :: a(3,4) = &
    reshape([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], shape(a))
  real(dp), parameter :: nAll = norm2(a)
  real(dp), parameter :: check_nAll = sqrt(sum(a * a))
  logical, parameter :: test_all = nAll == check_nAll
  real(dp), parameter :: norms1(4) = norm2(a, dim=1)
  real(dp), parameter :: check_norms1(4) = sqrt(sum(a * a, dim=1))
  logical, parameter :: test_norms1 = all(norms1 == check_norms1)
  real(dp), parameter :: norms2(3) = norm2(a, dim=2)
  real(dp), parameter :: check_norms2(3) = sqrt(sum(a * a, dim=2))
  logical, parameter :: test_norms2 = all(norms2 == check_norms2)
  logical, parameter :: test_normZ = norm2([0.,0.,0.]) == 0.
end
