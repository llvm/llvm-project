! RUN: %python %S/test_folding.py %s %flang_fc1 -pedantic
! Test intrinsic function folding edge case (both expected value and messages)
! These tests make assumptions regarding real(4) extrema.

#define TEST_ISNAN(v) logical, parameter :: test_##v =.NOT.(v.EQ.v)


module real_tests
  ! Test real(4) intrinsic folding on edge cases (inf and NaN)

  real(4), parameter :: r4_pmax = 3.4028235E38
  real(4), parameter :: r4_nmax = -3.4028235E38
  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan = 0._4/0._4
  !WARN: warning: division by zero
  real(4), parameter :: r4_pinf = 1._4/0._4
  !WARN: warning: division by zero
  real(4), parameter :: r4_ninf = -1._4/0._4

  !WARN: warning: argument is out of range [-1., 1.]
  real(4), parameter :: nan_r4_acos1 = acos(1.1)
  TEST_ISNAN(nan_r4_acos1)
  !WARN: warning: argument is out of range [-1., 1.]
  real(4), parameter :: nan_r4_acos2 = acos(r4_pmax)
  TEST_ISNAN(nan_r4_acos2)
  !WARN: warning: argument is out of range [-1., 1.]
  real(4), parameter :: nan_r4_acos3 = acos(r4_nmax)
  TEST_ISNAN(nan_r4_acos3)
  !WARN: warning: argument is out of range [-1., 1.]
  real(4), parameter :: nan_r4_acos4 = acos(r4_ninf)
  TEST_ISNAN(nan_r4_acos4)
  !WARN: warning: argument is out of range [-1., 1.]
  real(4), parameter :: nan_r4_acos5 = acos(r4_pinf)
  TEST_ISNAN(nan_r4_acos5)
  !WARN: warning: argument is out of range [-1., 1.]
  real(8), parameter :: nan_r8_dasin1 = dasin(-1.1_8)
  TEST_ISNAN(nan_r8_dasin1)
  !WARN: warning: complex argument must be different from zero
  complex(4), parameter :: c4_clog1 = clog((0., 0.))
  !WARN: warning: MOD: P argument is zero
  real(4), parameter :: nan_r4_mod = mod(3.5, 0.)
  TEST_ISNAN(nan_r4_mod)
  real(4), parameter :: ok_r4_gamma = gamma(-1.1)
  !WARN: warning: argument must not be a negative integer or zero
  real(4), parameter :: r4_gamma1 = gamma(0.)
  !WARN: warning: argument must not be a negative integer or zero
  real(4), parameter :: r4_gamma2 = gamma(-1.)
  real(4), parameter :: ok_r4_log_gamma = log_gamma(-2.001)
  !WARN: warning: argument must not be a negative integer or zero
  real(4), parameter :: r4_log_gamma1 = log_gamma(0.)
  !WARN: warning: argument must not be a negative integer or zero
  real(4), parameter :: r4_log_gamma2 = log_gamma(-100001.)
  !WARN: warning: 'x' and 'y' arguments must not be both zero
  real(4), parameter :: r4_atan2 = atan2(0., 0.)

  !WARN: warning: overflow on evaluation of intrinsic function or operation
  logical, parameter :: test_exp_overflow = exp(256._4).EQ.r4_pinf
 contains
  subroutine s1(a,j)
    !WARN: warning: MOD: P argument is zero
    print *, mod(a, 0.)
    !WARN: warning: MODULO: P argument is zero
    print *, modulo(a, 0.)
    !WARN: warning: MOD: P argument is zero
    print *, mod(j, 0.)
    !WARN: warning: MODULO: P argument is zero
    print *, modulo(j, 0.)
  end
end module

module parentheses
  ! Test parentheses in folding (they are kept around constants to keep the
  ! distinction between variable and expressions and require special care).
  real(4), parameter :: x_nop = 0.1_4
  real(4), parameter :: x_p = (x_nop)
  logical, parameter :: test_parentheses1 = acos(x_p).EQ.acos(x_nop)
end module

module specific_extremums
  ! f18 accepts all type kinds for the arguments of specific extremum intrinsics
  ! instead of of only default kind (or double precision for DMAX1 and DMIN1).
  ! This extensions is implemented by using the related generic intrinsic and
  ! converting the result.
  ! The tests below are cases where an implementation that converts the arguments to the
  ! standard required types instead would give different results than the implementation
  ! specified for f18 (converting the result).
  integer(8), parameter :: max_i32_8 = 2_8**31-1
  integer, parameter :: expected_min0 = int(min(max_i32_8, 2_8*max_i32_8), 4)
  !WARN: portability: Argument types do not match specific intrinsic 'min0' requirements; using 'min' generic instead and converting the result to INTEGER(4) if needed
  integer, parameter :: result_min0 =  min0(max_i32_8, 2_8*max_i32_8)
  ! result_min0 would be -2  if arguments were converted to default integer.
  logical, parameter :: test_min0 = expected_min0 .EQ. result_min0

  real, parameter :: expected_amax0 = real(max(max_i32_8, 2_8*max_i32_8), 4)
  !WARN: portability: Argument types do not match specific intrinsic 'amax0' requirements; using 'max' generic instead and converting the result to REAL(4) if needed
  real, parameter :: result_amax0 = amax0(max_i32_8, 2_8*max_i32_8)
  ! result_amax0 would be 2.1474836E+09 if arguments were converted to default integer first.
  logical, parameter :: test_amax0 = expected_amax0 .EQ. result_amax0
end module
