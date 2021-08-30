! RUN: %S/test_folding.sh %s %t %flang_fc1
! REQUIRES: shell
! Tests folding of SQRT()
module m
  implicit none
  ! +Inf
  real(8), parameter :: inf8 = z'7ff0000000000000'
  logical, parameter :: test_inf8 = sqrt(inf8) == inf8
  ! max finite
  real(8), parameter :: h8 = huge(1.0_8), h8z = z'7fefffffffffffff'
  logical, parameter :: test_h8 = h8 == h8z
  real(8), parameter :: sqrt_h8 = sqrt(h8), sqrt_h8z = z'5fefffffffffffff'
  logical, parameter :: test_sqrt_h8 = sqrt_h8 == sqrt_h8z
  real(8), parameter :: sqr_sqrt_h8 = sqrt_h8 * sqrt_h8, sqr_sqrt_h8z = z'7feffffffffffffe'
  logical, parameter :: test_sqr_sqrt_h8 = sqr_sqrt_h8 == sqr_sqrt_h8z
  ! -0 (sqrt is -0)
  real(8), parameter :: n08 = z'8000000000000000'
  real(8), parameter :: sqrt_n08 = sqrt(n08)
!WARN: division by zero
  real(8), parameter :: inf_n08 = 1.0_8 / sqrt_n08, inf_n08z = z'fff0000000000000'
  logical, parameter :: test_n08 = inf_n08 == inf_n08z
  ! min normal
  real(8), parameter :: t8 = tiny(1.0_8), t8z = z'0010000000000000'
  logical, parameter :: test_t8 = t8 == t8z
  real(8), parameter :: sqrt_t8 = sqrt(t8), sqrt_t8z = z'2000000000000000'
  logical, parameter :: test_sqrt_t8 = sqrt_t8 == sqrt_t8z
  real(8), parameter :: sqr_sqrt_t8 = sqrt_t8 * sqrt_t8
  logical, parameter :: test_sqr_sqrt_t8 = sqr_sqrt_t8 == t8
  ! max subnormal
  real(8), parameter :: maxs8 = z'000fffffffffffff'
  real(8), parameter :: sqrt_maxs8 = sqrt(maxs8), sqrt_maxs8z = z'2000000000000000'
  logical, parameter :: test_sqrt_maxs8 = sqrt_maxs8 == sqrt_maxs8z
  ! min subnormal
  real(8), parameter :: mins8 = z'1'
  real(8), parameter :: sqrt_mins8 = sqrt(mins8), sqrt_mins8z = z'1e60000000000000'
  logical, parameter :: test_sqrt_mins8 = sqrt_mins8 == sqrt_mins8z
  real(8), parameter :: sqr_sqrt_mins8 = sqrt_mins8 * sqrt_mins8
  logical, parameter :: test_sqr_sqrt_mins8 = sqr_sqrt_mins8 == mins8
end module

