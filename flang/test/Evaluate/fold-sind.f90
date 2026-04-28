! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SIND() via rewrite to SIN(x * pi/180).
module m
  ! Kind preservation.
  logical, parameter :: test_kind4 = kind(sind(1.0_4)) == 4
  logical, parameter :: test_kind8 = kind(sind(1.0_8)) == 8

  ! sin(0) is exactly 0 in any reasonable libm.
  logical, parameter :: test_zero4 = sind(0.0_4) == 0.0_4
  logical, parameter :: test_zero8 = sind(0.0_8) == 0.0_8

  ! Tolerance-based checks tolerate small host libm differences.
  real(4), parameter :: res_30_4 = sind(30.0_4)
  real(8), parameter :: res_30_8 = sind(30.0_8)
  logical, parameter :: test_30_4 = abs(res_30_4 - 0.5_4) <= 1.0e-6_4
  logical, parameter :: test_30_8 = abs(res_30_8 - 0.5_8) <= 1.0e-12_8

  real(4), parameter :: res_90_4 = sind(90.0_4)
  real(8), parameter :: res_90_8 = sind(90.0_8)
  logical, parameter :: test_90_4 = abs(res_90_4 - 1.0_4) <= 1.0e-6_4
  logical, parameter :: test_90_8 = abs(res_90_8 - 1.0_8) <= 1.0e-12_8

  real(4), parameter :: res_neg90_4 = sind(-90.0_4)
  logical, parameter :: test_neg90_4 = abs(res_neg90_4 + 1.0_4) <= 1.0e-6_4

  ! Elemental application over an array argument also folds.
  logical, parameter :: test_array = &
      all(abs(sind([0.0_4, 90.0_4]) - [0.0_4, 1.0_4]) <= 1.0e-6_4)
end
