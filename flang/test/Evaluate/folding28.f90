! RUN: %python %S/test_folding.py %s %flang_fc1
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
  !WARN: warning: division by zero
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
  real(8), parameter :: sqrt_maxs8 = sqrt(maxs8), sqrt_maxs8z = z'1fffffffffffffff'
  logical, parameter :: test_sqrt_maxs8 = sqrt_maxs8 == sqrt_maxs8z
  ! min subnormal
  real(8), parameter :: mins8 = z'1'
  real(8), parameter :: sqrt_mins8 = sqrt(mins8), sqrt_mins8z = z'1e60000000000000'
  logical, parameter :: test_sqrt_mins8 = sqrt_mins8 == sqrt_mins8z
  real(8), parameter :: sqr_sqrt_mins8 = sqrt_mins8 * sqrt_mins8
  logical, parameter :: test_sqr_sqrt_mins8 = sqr_sqrt_mins8 == mins8
  ! regression tests: cases near 1.
  real(4), parameter :: sqrt_under1 = sqrt(.96875)
  logical, parameter :: test_sqrt_under1 = sqrt_under1 == .984250962734222412109375
  ! oddball case: the value before 1. is also its own sqrt, but not its own square
  real(4), parameter :: before_1 = z'3f7fffff' ! .999999940395355224609375
  real(4), parameter :: sqrt_before_1 = sqrt(before_1)
  logical, parameter :: test_before_1 = sqrt_before_1 == before_1
  real(4), parameter :: sq_sqrt_before_1 = sqrt_before_1 * sqrt_before_1
  logical, parameter :: test_sq_before_1 = sq_sqrt_before_1 < before_1
  ! ICE at 0.0
  real(4), parameter :: sqrt_zero_4 = sqrt(0.0)
  logical, parameter :: test_sqrt_zero_4 = sqrt_zero_4 == 0.0
  real(8), parameter :: sqrt_zero_8 = sqrt(0.0)
  logical, parameter :: test_sqrt_zero_8 = sqrt_zero_8 == 0.0
  ! Some common values to get right
  real(8), parameter :: sqrt_1_8 = sqrt(1.d0)
  logical, parameter :: test_sqrt_1_8 = sqrt_1_8 == 1.d0
  real(8), parameter :: sqrt_2_8 = sqrt(2.d0)
  logical, parameter :: test_sqrt_2_8 = sqrt_2_8 == 1.4142135623730951454746218587388284504413604736328125d0
  real(8), parameter :: sqrt_3_8 = sqrt(3.d0)
  logical, parameter :: test_sqrt_3_8 = sqrt_3_8 == 1.732050807568877193176604123436845839023590087890625d0
  real(8), parameter :: sqrt_4_8 = sqrt(4.d0)
  logical, parameter :: test_sqrt_4_8 = sqrt_4_8 == 2.d0
  real(8), parameter :: sqrt_5_8 = sqrt(5.d0)
  logical, parameter :: test_sqrt_5_8 = sqrt_5_8 == 2.236067977499789805051477742381393909454345703125d0
  real(8), parameter :: sqrt_6_8 = sqrt(6.d0)
  logical, parameter :: test_sqrt_6_8 = sqrt_6_8 == 2.44948974278317788133563226438127458095550537109375d0
  real(8), parameter :: sqrt_7_8 = sqrt(7.d0)
  logical, parameter :: test_sqrt_7_8 = sqrt_7_8 == 2.64575131106459071617109657381661236286163330078125d0
  real(8), parameter :: sqrt_8_8 = sqrt(8.d0)
  logical, parameter :: test_sqrt_8_8 = sqrt_8_8 == 2.828427124746190290949243717477656900882720947265625d0
  real(8), parameter :: sqrt_9_8 = sqrt(9.d0)
  logical, parameter :: test_sqrt_9_8 = sqrt_9_8 == 3.d0
  real(8), parameter :: sqrt_10_8 = sqrt(10.d0)
  logical, parameter :: test_sqrt_10_8 = sqrt_10_8 == 3.162277660168379522787063251598738133907318115234375d0
end module
