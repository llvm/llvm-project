! Test intrinsic function folding edge case (both expected value and messages)
! These tests make assumptions regarding real(4) extrema.

#define TEST_ISNAN(v) logical, parameter :: test_##v =.NOT.(v.EQ.v)


module real_tests
  ! Test real(4) intrinsic folding on edge cases (inf and NaN)

  real(4), parameter :: r4_pmax = 3.4028235E38
  real(4), parameter :: r4_nmax = -3.4028235E38
  !WARN: invalid argument on division
  real(4), parameter :: r4_nan = 0._4/0._4
  !WARN: division by zero on division
  real(4), parameter :: r4_pinf = 1._4/0._4
  !WARN: division by zero on division
  real(4), parameter :: r4_ninf = -1._4/0._4

  !WARN: invalid argument on intrinsic function
  real(4), parameter :: nan_r4_acos1 = acos(1.1)
  TEST_ISNAN(nan_r4_acos1)
  !WARN: invalid argument on intrinsic function
  real(4), parameter :: nan_r4_acos2 = acos(r4_pmax)
  TEST_ISNAN(nan_r4_acos2)
  !WARN: invalid argument on intrinsic function
  real(4), parameter :: nan_r4_acos3 = acos(r4_nmax)
  TEST_ISNAN(nan_r4_acos3)
  !WARN: invalid argument on intrinsic function
  real(4), parameter :: nan_r4_acos4 = acos(r4_ninf)
  TEST_ISNAN(nan_r4_acos4)
  !WARN: invalid argument on intrinsic function
  real(4), parameter :: nan_r4_acos5 = acos(r4_pinf)
  TEST_ISNAN(nan_r4_acos5)

  !WARN: overflow on intrinsic function
  logical, parameter :: test_exp_overflow = exp(256._4).EQ.r4_pinf
end module

module parentheses
  ! Test parentheses in folding (they are kept around constants to keep the
  ! distinction between variable and expressions and require special care).
  real(4), parameter :: x_nop = 0.1_4
  real(4), parameter :: x_p = (x_nop)
  logical, parameter :: test_parentheses1 = acos(x_p).EQ.acos(x_nop)
end module
