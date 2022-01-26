!===-- module/ieee_arithmetic.f90 ------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2018, clause 17.2
module ieee_arithmetic

  use __Fortran_builtins, only: &
    ieee_is_nan => __builtin_ieee_is_nan, &
    ieee_is_normal => __builtin_ieee_is_normal, &
    ieee_is_negative => __builtin_ieee_is_negative, &
    ieee_next_after => __builtin_ieee_next_after, &
    ieee_next_down => __builtin_ieee_next_down, &
    ieee_next_up => __builtin_ieee_next_up, &
    ieee_scalb => scale, &
    ieee_selected_real_kind => __builtin_ieee_selected_real_kind, &
    ieee_support_datatype => __builtin_ieee_support_datatype, &
    ieee_support_denormal => __builtin_ieee_support_denormal, &
    ieee_support_divide => __builtin_ieee_support_divide, &
    ieee_support_inf => __builtin_ieee_support_inf, &
    ieee_support_io => __builtin_ieee_support_io, &
    ieee_support_nan => __builtin_ieee_support_nan, &
    ieee_support_sqrt => __builtin_ieee_support_sqrt, &
    ieee_support_standard => __builtin_ieee_support_standard, &
    ieee_support_subnormal => __builtin_ieee_support_subnormal, &
    ieee_support_underflow_control => __builtin_ieee_support_underflow_control

  implicit none

  type :: ieee_class_type
    private
    integer(kind=1) :: which = 0
  end type ieee_class_type

  type(ieee_class_type), parameter :: &
    ieee_signaling_nan = ieee_class_type(1), &
    ieee_quiet_nan = ieee_class_type(2), &
    ieee_negative_inf = ieee_class_type(3), &
    ieee_negative_normal = ieee_class_type(4), &
    ieee_negative_denormal = ieee_class_type(5), &
    ieee_negative_zero = ieee_class_type(6), &
    ieee_positive_zero = ieee_class_type(7), &
    ieee_positive_subnormal = ieee_class_type(8), &
    ieee_positive_normal = ieee_class_type(9), &
    ieee_positive_inf = ieee_class_type(10), &
    ieee_other_value = ieee_class_type(11)

  type(ieee_class_type), parameter :: &
    ieee_negative_subnormal = ieee_negative_denormal, &
    ieee_positive_denormal = ieee_negative_subnormal

  type :: ieee_round_type
    private
    integer(kind=1) :: mode = 0
  end type ieee_round_type

  type(ieee_round_type), parameter :: &
    ieee_nearest = ieee_round_type(1), &
    ieee_to_zero = ieee_round_type(2), &
    ieee_up = ieee_round_type(3), &
    ieee_down = ieee_round_type(4), &
    ieee_away = ieee_round_type(5), &
    ieee_other = ieee_round_type(6)

  interface operator(==)
    module procedure class_eq
    module procedure round_eq
  end interface operator(==)
  interface operator(/=)
    module procedure class_ne
    module procedure round_ne
  end interface operator(/=)
  private :: class_eq, class_ne, round_eq, round_ne

  ! See Fortran 2018, 17.10 & 17.11
  generic :: ieee_class => ieee_class_a2, ieee_class_a3, ieee_class_a4, ieee_class_a8, ieee_class_a10, ieee_class_a16
  private :: ieee_class_a2, ieee_class_a3, ieee_class_a4, ieee_class_a8, ieee_class_a10, ieee_class_a16

  generic :: ieee_copy_sign => ieee_copy_sign_a2, ieee_copy_sign_a3, ieee_copy_sign_a4, ieee_copy_sign_a8, ieee_copy_sign_a10, ieee_copy_sign_a16
  private :: ieee_copy_sign_a2, ieee_copy_sign_a3, ieee_copy_sign_a4, ieee_copy_sign_a8, ieee_copy_sign_a10, ieee_copy_sign_a16

  generic :: ieee_is_finite => ieee_is_finite_a2, ieee_is_finite_a3, ieee_is_finite_a4, ieee_is_finite_a8, ieee_is_finite_a10, ieee_is_finite_a16
  private :: ieee_is_finite_a2, ieee_is_finite_a3, ieee_is_finite_a4, ieee_is_finite_a8, ieee_is_finite_a10, ieee_is_finite_a16

  generic :: ieee_rem => &
    ieee_rem_a2_a2, ieee_rem_a2_a3, ieee_rem_a2_a4, ieee_rem_a2_a8, ieee_rem_a2_a10, ieee_rem_a2_a16, &
    ieee_rem_a3_a2, ieee_rem_a3_a3, ieee_rem_a3_a4, ieee_rem_a3_a8, ieee_rem_a3_a10, ieee_rem_a3_a16, &
    ieee_rem_a4_a2, ieee_rem_a4_a3, ieee_rem_a4_a4, ieee_rem_a4_a8, ieee_rem_a4_a10, ieee_rem_a4_a16, &
    ieee_rem_a8_a2, ieee_rem_a8_a3, ieee_rem_a8_a4, ieee_rem_a8_a8, ieee_rem_a8_a10, ieee_rem_a8_a16, &
    ieee_rem_a10_a2, ieee_rem_a10_a3, ieee_rem_a10_a4, ieee_rem_a10_a8, ieee_rem_a10_a10, ieee_rem_a10_a16, &
    ieee_rem_a16_a2, ieee_rem_a16_a3, ieee_rem_a16_a4, ieee_rem_a16_a8, ieee_rem_a16_a10, ieee_rem_a16_a16
  private :: &
    ieee_rem_a2_a2, ieee_rem_a2_a3, ieee_rem_a2_a4, ieee_rem_a2_a8, ieee_rem_a2_a10, ieee_rem_a2_a16, &
    ieee_rem_a3_a2, ieee_rem_a3_a3, ieee_rem_a3_a4, ieee_rem_a3_a8, ieee_rem_a3_a10, ieee_rem_a3_a16, &
    ieee_rem_a4_a2, ieee_rem_a4_a3, ieee_rem_a4_a4, ieee_rem_a4_a8, ieee_rem_a4_a10, ieee_rem_a4_a16, &
    ieee_rem_a8_a2, ieee_rem_a8_a3, ieee_rem_a8_a4, ieee_rem_a8_a8, ieee_rem_a8_a10, ieee_rem_a8_a16, &
    ieee_rem_a10_a2, ieee_rem_a10_a3, ieee_rem_a10_a4, ieee_rem_a10_a8, ieee_rem_a10_a10, ieee_rem_a10_a16, &
    ieee_rem_a16_a2, ieee_rem_a16_a3, ieee_rem_a16_a4, ieee_rem_a16_a8, ieee_rem_a16_a10, ieee_rem_a16_a16

  generic :: ieee_support_rounding => ieee_support_rounding_, &
    ieee_support_rounding_2, ieee_support_rounding_3, &
    ieee_support_rounding_4, ieee_support_rounding_8, &
    ieee_support_rounding_10, ieee_support_rounding_16
  private :: ieee_support_rounding_, &
    ieee_support_rounding_2, ieee_support_rounding_3, &
    ieee_support_rounding_4, ieee_support_rounding_8, &
    ieee_support_rounding_10, ieee_support_rounding_16

  ! TODO: more interfaces (_fma, &c.)

  private :: classify

 contains

  elemental logical function class_eq(x,y)
    type(ieee_class_type), intent(in) :: x, y
    class_eq = x%which == y%which
  end function class_eq

  elemental logical function class_ne(x,y)
    type(ieee_class_type), intent(in) :: x, y
    class_ne = x%which /= y%which
  end function class_ne

  elemental logical function round_eq(x,y)
    type(ieee_round_type), intent(in) :: x, y
    round_eq = x%mode == y%mode
  end function round_eq

  elemental logical function round_ne(x,y)
    type(ieee_round_type), intent(in) :: x, y
    round_ne = x%mode /= y%mode
  end function round_ne

  elemental type(ieee_class_type) function classify( &
      expo,maxExpo,negative,significandNZ,quietBit)
    integer, intent(in) :: expo, maxExpo
    logical, intent(in) :: negative, significandNZ, quietBit
    if (expo == 0) then
      if (significandNZ) then
        if (negative) then
          classify = ieee_negative_denormal
        else
          classify = ieee_positive_denormal
        end if
      else
        if (negative) then
          classify = ieee_negative_zero
        else
          classify = ieee_positive_zero
        end if
      end if
    else if (expo == maxExpo) then
      if (significandNZ) then
        if (quietBit) then
          classify = ieee_quiet_nan
        else
          classify = ieee_signaling_nan
        end if
      else
        if (negative) then
          classify = ieee_negative_inf
        else
          classify = ieee_positive_inf
        end if
      end if
    else
      if (negative) then
        classify = ieee_negative_normal
      else
        classify = ieee_positive_normal
      end if
    end if
  end function classify

#define _CLASSIFY(RKIND,IKIND,TOTALBITS,PREC,IMPLICIT) \
  type(ieee_class_type) elemental function ieee_class_a##RKIND(x); \
    real(kind=RKIND), intent(in) :: x; \
    integer(kind=IKIND) :: raw; \
    integer, parameter :: significand = PREC - IMPLICIT; \
    integer, parameter :: exponentBits = TOTALBITS - 1 - significand; \
    integer, parameter :: maxExpo = shiftl(1, exponentBits) - 1; \
    integer :: exponent, sign; \
    logical :: negative, nzSignificand, quiet; \
    raw = transfer(x, raw); \
    exponent = ibits(raw, significand, exponentBits); \
    negative = btest(raw, TOTALBITS - 1); \
    nzSignificand = ibits(raw, 0, significand) /= 0; \
    quiet = btest(raw, significand - 1); \
    ieee_class_a##RKIND = classify(exponent, maxExpo, negative, nzSignificand, quiet); \
  end function ieee_class_a##RKIND
  _CLASSIFY(2,2,16,11,1)
  _CLASSIFY(3,2,16,8,1)
  _CLASSIFY(4,4,32,24,1)
  _CLASSIFY(8,8,64,53,1)
  _CLASSIFY(10,16,80,64,0)
  _CLASSIFY(16,16,128,112,1)
#undef _CLASSIFY

  ! TODO: This might need to be an actual Operation instead
#define _COPYSIGN(RKIND,IKIND,BITS) \
  real(kind=RKIND) elemental function ieee_copy_sign_a##RKIND(x,y); \
    real(kind=RKIND), intent(in) :: x, y; \
    integer(kind=IKIND) :: xbits, ybits; \
    xbits = transfer(x, xbits); \
    ybits = transfer(y, ybits); \
    xbits = ior(ibclr(xbits, BITS-1), iand(ybits, shiftl(1_##IKIND, BITS-1))); \
    ieee_copy_sign_a##RKIND = transfer(xbits, x); \
  end function ieee_copy_sign_a##RKIND
  _COPYSIGN(2,2,16)
  _COPYSIGN(3,2,16)
  _COPYSIGN(4,4,32)
  _COPYSIGN(8,8,64)
  _COPYSIGN(10,16,80)
  _COPYSIGN(16,16,128)
#undef _COPYSIGN

#define _IS_FINITE(KIND) \
  elemental function ieee_is_finite_a##KIND(x) result(res); \
    real(kind=KIND), intent(in) :: x; \
    logical :: res; \
    type(ieee_class_type) :: classification; \
    classification = ieee_class(x); \
    res = classification == ieee_negative_zero .or. classification == ieee_positive_zero \
     .or. classification == ieee_negative_denormal .or. classification == ieee_positive_denormal \
     .or. classification == ieee_negative_normal .or. classification == ieee_positive_normal; \
  end function
  _IS_FINITE(2)
  _IS_FINITE(3)
  _IS_FINITE(4)
  _IS_FINITE(8)
  _IS_FINITE(10)
  _IS_FINITE(16)
#undef _IS_FINITE

#define _IS_NEGATIVE(KIND) \
  elemental function ieee_is_negative_a##KIND(x) result(res); \
    real(kind=KIND), intent(in) :: x; \
    logical :: res; \
    type(ieee_class_type) :: classification; \
    classification = ieee_class(x); \
    res = classification == ieee_negative_zero .or. classification == ieee_negative_denormal \
     .or. classification == ieee_negative_normal .or. classification == ieee_negative_inf; \
  end function
  _IS_NEGATIVE(2)
  _IS_NEGATIVE(3)
  _IS_NEGATIVE(4)
  _IS_NEGATIVE(8)
  _IS_NEGATIVE(10)
  _IS_NEGATIVE(16)
#undef _IS_NEGATIVE

#define _IS_NORMAL(KIND) \
  elemental function ieee_is_normal_a##KIND(x) result(res); \
    real(kind=KIND), intent(in) :: x; \
    logical :: res; \
    type(ieee_class_type) :: classification; \
    classification = ieee_class(x); \
    res = classification == ieee_negative_normal .or. classification == ieee_positive_normal \
      .or. classification == ieee_negative_zero .or. classification == ieee_positive_zero; \
  end function
  _IS_NORMAL(2)
  _IS_NORMAL(3)
  _IS_NORMAL(4)
  _IS_NORMAL(8)
  _IS_NORMAL(10)
  _IS_NORMAL(16)
#undef _IS_NORMAL

! TODO: handle edge cases from 17.11.31
#define _REM(XKIND,YKIND) \
  elemental function ieee_rem_a##XKIND##_a##YKIND(x, y) result(res); \
    real(kind=XKIND), intent(in) :: x; \
    real(kind=YKIND), intent(in) :: y; \
    integer, parameter :: rkind = max(XKIND, YKIND); \
    real(kind=rkind) :: res, tmp; \
    tmp = anint(real(x, kind=rkind) / y); \
    res = x - y * tmp; \
  end function
  _REM(2,2)
  _REM(2,3)
  _REM(2,4)
  _REM(2,8)
  _REM(2,10)
  _REM(2,16)
  _REM(3,2)
  _REM(3,3)
  _REM(3,4)
  _REM(3,8)
  _REM(3,10)
  _REM(3,16)
  _REM(4,2)
  _REM(4,3)
  _REM(4,4)
  _REM(4,8)
  _REM(4,10)
  _REM(4,16)
  _REM(8,2)
  _REM(8,3)
  _REM(8,4)
  _REM(8,8)
  _REM(8,10)
  _REM(8,16)
  _REM(10,2)
  _REM(10,3)
  _REM(10,4)
  _REM(10,8)
  _REM(10,10)
  _REM(10,16)
  _REM(16,2)
  _REM(16,3)
  _REM(16,4)
  _REM(16,8)
  _REM(16,10)
  _REM(16,16)
#undef _REM

  pure logical function ieee_support_rounding_(round_type)
    type(ieee_round_type), intent(in) :: round_type
    ieee_support_rounding_ = .true.
  end function
  pure logical function ieee_support_rounding_2(round_type,x)
    type(ieee_round_type), intent(in) :: round_type
    real(kind=2), intent(in) :: x
    ieee_support_rounding_2 = .true.
  end function
  pure logical function ieee_support_rounding_3(round_type,x)
    type(ieee_round_type), intent(in) :: round_type
    real(kind=3), intent(in) :: x
    ieee_support_rounding_3 = .true.
  end function
  pure logical function ieee_support_rounding_4(round_type,x)
    type(ieee_round_type), intent(in) :: round_type
    real(kind=4), intent(in) :: x
    ieee_support_rounding_4 = .true.
  end function
  pure logical function ieee_support_rounding_8(round_type,x)
    type(ieee_round_type), intent(in) :: round_type
    real(kind=8), intent(in) :: x
    ieee_support_rounding_8 = .true.
  end function
  pure logical function ieee_support_rounding_10(round_type,x)
    type(ieee_round_type), intent(in) :: round_type
    real(kind=10), intent(in) :: x
    ieee_support_rounding_10 = .true.
  end function
  pure logical function ieee_support_rounding_16(round_type,x)
    type(ieee_round_type), intent(in) :: round_type
    real(kind=16), intent(in) :: x
    ieee_support_rounding_16 = .true.
  end function

end module ieee_arithmetic
