!===-- module/ieee_arithmetic.f90 ------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! Fortran 2018 Clause 17

module ieee_arithmetic
  ! 17.1: "The module IEEE_ARITHMETIC behaves as if it contained a
  ! USE statement for IEEE_EXCEPTIONS; everything that is public in
  ! IEEE_EXCEPTIONS is public in IEEE_ARITHMETIC."
  use __Fortran_ieee_exceptions

  use __Fortran_builtins, only: &
    ieee_is_nan => __builtin_ieee_is_nan, &
    ieee_is_negative => __builtin_ieee_is_negative, &
    ieee_is_normal => __builtin_ieee_is_normal, &
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
    elemental logical function ieee_class_eq(x, y)
      import ieee_class_type
      type(ieee_class_type), intent(in) :: x, y
    end function ieee_class_eq
    elemental logical function ieee_round_eq(x, y)
      import ieee_round_type
      type(ieee_round_type), intent(in) :: x, y
    end function ieee_round_eq
  end interface operator(==)
  interface operator(/=)
    elemental logical function ieee_class_ne(x, y)
      import ieee_class_type
      type(ieee_class_type), intent(in) :: x, y
    end function ieee_class_ne
    elemental logical function ieee_round_ne(x, y)
      import ieee_round_type
      type(ieee_round_type), intent(in) :: x, y
    end function ieee_round_ne
  end interface operator(/=)
  private :: ieee_class_eq, ieee_round_eq, ieee_class_ne, ieee_round_ne

! Define specifics with 1 or 2 INTEGER, LOGICAL, or REAL arguments for
! generic G.
#define SPECIFICS_I(G) \
  G(1) G(2) G(4) G(8) G(16)
#define SPECIFICS_L(G) \
  G(1) G(2) G(4) G(8)
#define SPECIFICS_R(G) \
  G(2) G(3) G(4) G(8) G(10) G(16)
#define SPECIFICS_II(G) \
  G(1,1) G(1,2) G(1,4) G(1,8) G(1,16) \
  G(2,1) G(2,2) G(2,4) G(2,8) G(2,16) \
  G(4,1) G(4,2) G(4,4) G(4,8) G(4,16) \
  G(8,1) G(8,2) G(8,4) G(8,8) G(8,16) \
  G(16,1) G(16,2) G(16,4) G(16,8) G(16,16)
#define SPECIFICS_RI(G) \
  G(2,1) G(2,2) G(2,4) G(2,8) G(2,16) \
  G(3,1) G(3,2) G(3,4) G(3,8) G(3,16) \
  G(4,1) G(4,2) G(4,4) G(4,8) G(4,16) \
  G(8,1) G(8,2) G(8,4) G(8,8) G(8,16) \
  G(10,1) G(10,2) G(10,4) G(10,8) G(10,16) \
  G(16,1) G(16,2) G(16,4) G(16,8) G(16,16)
#define SPECIFICS_RR(G) \
  G(2,2) G(2,3) G(2,4) G(2,8) G(2,10) G(2,16) \
  G(3,2) G(3,3) G(3,4) G(3,8) G(3,10) G(3,16) \
  G(4,2) G(4,3) G(4,4) G(4,8) G(4,10) G(4,16) \
  G(8,2) G(8,3) G(8,4) G(8,8) G(8,10) G(8,16) \
  G(10,2) G(10,3) G(10,4) G(10,8) G(10,10) G(10,16) \
  G(16,2) G(16,3) G(16,4) G(16,8) G(16,10) G(16,16)

! Set PRIVATE accessibility for specifics with 1 or 2 INTEGER, LOGICAL, or REAL
! arguments for generic G.
#define PRIVATE_I(G) private :: \
  G##_i1, G##_i2, G##_i4, G##_i8, G##_i16
#define PRIVATE_L(G) private :: \
  G##_l1, G##_l2, G##_l4, G##_l8
#define PRIVATE_R(G) private :: \
  G##_a2, G##_a3, G##_a4, G##_a8, G##_a10, G##_a16
#define PRIVATE_II(G) private :: \
  G##_i1_i1, G##_i1_i2, G##_i1_i4, G##_i1_i8, G##_i1_i16, \
  G##_i2_i1, G##_i2_i2, G##_i2_i4, G##_i2_i8, G##_i2_i16, \
  G##_i4_i1, G##_i4_i2, G##_i4_i4, G##_i4_i8, G##_i4_i16, \
  G##_i8_i1, G##_i8_i2, G##_i8_i4, G##_i8_i8, G##_i8_i16, \
  G##_i16_i1, G##_i16_i2, G##_i16_i4, G##_i16_i8, G##_i16_i16
#define PRIVATE_RI(G) private :: \
  G##_a2_i1, G##_a2_i2, G##_a2_i4, G##_a2_i8, G##_a2_i16, \
  G##_a3_i1, G##_a3_i2, G##_a3_i4, G##_a3_i8, G##_a3_i16, \
  G##_a4_i1, G##_a4_i2, G##_a4_i4, G##_a4_i8, G##_a4_i16, \
  G##_a8_i1, G##_a8_i2, G##_a8_i4, G##_a8_i8, G##_a8_i16, \
  G##_a10_i1, G##_a10_i2, G##_a10_i4, G##_a10_i8, G##_a10_i16, \
  G##_a16_i1, G##_a16_i2, G##_a16_i4, G##_a16_i8, G##_a16_i16
#define PRIVATE_RR(G) private :: \
  G##_a2_a2, G##_a2_a3, G##_a2_a4, G##_a2_a8, G##_a2_a10, G##_a2_a16, \
  G##_a3_a2, G##_a3_a3, G##_a3_a4, G##_a3_a8, G##_a3_a10, G##_a3_a16, \
  G##_a4_a2, G##_a4_a3, G##_a4_a4, G##_a4_a8, G##_a4_a10, G##_a4_a16, \
  G##_a8_a2, G##_a8_a3, G##_a8_a4, G##_a8_a8, G##_a8_a10, G##_a8_a16, \
  G##_a10_a2, G##_a10_a3, G##_a10_a4, G##_a10_a8, G##_a10_a10, G##_a10_a16, \
  G##_a16_a2, G##_a16_a3, G##_a16_a4, G##_a16_a8, G##_a16_a10, G##_a16_a16

#define IEEE_CLASS_R(XKIND) \
  elemental type(ieee_class_type) function ieee_class_a##XKIND(x); \
    import ieee_class_type; \
    real(XKIND), intent(in) :: x; \
  end function ieee_class_a##XKIND;
  interface ieee_class
    SPECIFICS_R(IEEE_CLASS_R)
  end interface ieee_class
  PRIVATE_R(IEEE_CLASS)
#undef IEEE_CLASS_R

#define IEEE_COPY_SIGN_RR(XKIND, YKIND) \
  elemental real(XKIND) function ieee_copy_sign_a##XKIND##_a##YKIND(x, y); \
    real(XKIND), intent(in) :: x; \
    real(YKIND), intent(in) :: y; \
  end function ieee_copy_sign_a##XKIND##_a##YKIND;
  interface ieee_copy_sign
    SPECIFICS_RR(IEEE_COPY_SIGN_RR)
  end interface ieee_copy_sign
  PRIVATE_RR(IEEE_COPY_SIGN)
#undef IEEE_COPY_SIGN_RR

#define IEEE_FMA_R(AKIND) \
  elemental real(AKIND) function ieee_fma_a##AKIND(a, b, c); \
    real(AKIND), intent(in) :: a, b, c; \
  end function ieee_fma_a##AKIND;
  interface ieee_fma
    SPECIFICS_R(IEEE_FMA_R)
  end interface ieee_fma
  PRIVATE_R(IEEE_FMA)
#undef IEEE_FMA_R

#define IEEE_GET_ROUNDING_MODE_I(RKIND) \
  subroutine ieee_get_rounding_mode_i##RKIND(round_value, radix); \
    import ieee_round_type; \
    type(ieee_round_type), intent(out) :: round_value; \
    integer(RKIND), intent(in) :: radix; \
  end subroutine ieee_get_rounding_mode_i##RKIND;
  interface ieee_get_rounding_mode
    subroutine ieee_get_rounding_mode(round_value)
      import ieee_round_type
      type(ieee_round_type), intent(out) :: round_value
    end subroutine ieee_get_rounding_mode
    SPECIFICS_I(IEEE_GET_ROUNDING_MODE_I)
  end interface ieee_get_rounding_mode
  PRIVATE_I(IEEE_GET_ROUNDING_MODE)
#undef IEEE_GET_ROUNDING_MODE_I

#define IEEE_GET_UNDERFLOW_MODE_L(GKIND) \
  subroutine ieee_get_underflow_mode_l##GKIND(gradual); \
    logical(GKIND), intent(out) :: gradual; \
  end subroutine ieee_get_underflow_mode_l##GKIND;
  interface ieee_get_underflow_mode
    SPECIFICS_L(IEEE_GET_UNDERFLOW_MODE_L)
  end interface ieee_get_underflow_mode
  PRIVATE_L(IEEE_GET_UNDERFLOW_MODE)
#undef IEEE_GET_UNDERFLOW_MODE_L

! When kind argument is present, kind(result) is value(kind), not kind(kind).
! That is not known here, so return integer(16).
#define IEEE_INT_R(AKIND) \
  elemental integer function ieee_int_a##AKIND(a, round); \
    import ieee_round_type; \
    real(AKIND), intent(in) :: a; \
    type(ieee_round_type), intent(in) :: round; \
  end function ieee_int_a##AKIND;
#define IEEE_INT_RI(AKIND, KKIND) \
  elemental integer(16) function ieee_int_a##AKIND##_i##KKIND(a, round, kind); \
    import ieee_round_type; \
    real(AKIND), intent(in) :: a; \
    type(ieee_round_type), intent(in) :: round; \
    integer(KKIND), intent(in) :: kind; \
  end function ieee_int_a##AKIND##_i##KKIND;
  interface ieee_int
    SPECIFICS_R(IEEE_INT_R)
    SPECIFICS_RI(IEEE_INT_RI)
  end interface ieee_int
  PRIVATE_R(IEEE_INT)
  PRIVATE_RI(IEEE_INT)
#undef IEEE_INT_R
#undef IEEE_INT_RI

#define IEEE_IS_FINITE_R(XKIND) \
  elemental logical function ieee_is_finite_a##XKIND(x); \
    real(XKIND), intent(in) :: x; \
  end function ieee_is_finite_a##XKIND;
  interface ieee_is_finite
    SPECIFICS_R(IEEE_IS_FINITE_R)
  end interface ieee_is_finite
  PRIVATE_R(IEEE_IS_FINITE)
#undef IEEE_IS_FINITE_R

#define IEEE_LOGB_R(XKIND) \
  elemental real(XKIND) function ieee_logb_a##XKIND(x); \
    real(XKIND), intent(in) :: x; \
  end function ieee_logb_a##XKIND;
  interface ieee_logb
    SPECIFICS_R(IEEE_LOGB_R)
  end interface ieee_logb
  PRIVATE_R(IEEE_LOGB)
#undef IEEE_LOGB_R

#define IEEE_MAX_NUM_R(XKIND) \
  elemental real(XKIND) function ieee_max_num_a##XKIND(x, y); \
    real(XKIND), intent(in) :: x, y; \
  end function ieee_max_num_a##XKIND;
  interface ieee_max_num
    SPECIFICS_R(IEEE_MAX_NUM_R)
  end interface ieee_max_num
  PRIVATE_R(IEEE_MAX_NUM)
#undef IEEE_MAX_NUM_R

#define IEEE_MAX_NUM_MAG_R(XKIND) \
  elemental real(XKIND) function ieee_max_num_mag_a##XKIND(x, y); \
    real(XKIND), intent(in) :: x, y; \
  end function ieee_max_num_mag_a##XKIND;
  interface ieee_max_num_mag
    SPECIFICS_R(IEEE_MAX_NUM_MAG_R)
  end interface ieee_max_num_mag
  PRIVATE_R(IEEE_MAX_NUM_MAG)
#undef IEEE_MAX_NUM_MAG_R

#define IEEE_MIN_NUM_R(XKIND) \
  elemental real(XKIND) function ieee_min_num_a##XKIND(x, y); \
    real(XKIND), intent(in) :: x, y; \
  end function ieee_min_num_a##XKIND;
  interface ieee_min_num
    SPECIFICS_R(IEEE_MIN_NUM_R)
  end interface ieee_min_num
  PRIVATE_R(IEEE_MIN_NUM)
#undef IEEE_MIN_NUM_R

#define IEEE_MIN_NUM_MAG_R(XKIND) \
  elemental real(XKIND) function ieee_min_num_mag_a##XKIND(x, y); \
    real(XKIND), intent(in) :: x, y; \
  end function ieee_min_num_mag_a##XKIND;
  interface ieee_min_num_mag
    SPECIFICS_R(IEEE_MIN_NUM_MAG_R)
  end interface ieee_min_num_mag
  PRIVATE_R(IEEE_MIN_NUM_MAG)
#undef IEEE_MIN_NUM_MAG_R

#define IEEE_QUIET_EQ_R(AKIND) \
  elemental logical function ieee_quiet_eq_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_quiet_eq_a##AKIND;
  interface ieee_quiet_eq
    SPECIFICS_R(IEEE_QUIET_EQ_R)
  end interface ieee_quiet_eq
  PRIVATE_R(IEEE_QUIET_EQ)
#undef IEEE_QUIET_EQ_R

#define IEEE_QUIET_GE_R(AKIND) \
  elemental logical function ieee_quiet_ge_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_quiet_ge_a##AKIND;
  interface ieee_quiet_ge
    SPECIFICS_R(IEEE_QUIET_GE_R)
  end interface ieee_quiet_ge
  PRIVATE_R(IEEE_QUIET_GE)
#undef IEEE_QUIET_GE_R

#define IEEE_QUIET_GT_R(AKIND) \
  elemental logical function ieee_quiet_gt_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_quiet_gt_a##AKIND;
  interface ieee_quiet_gt
    SPECIFICS_R(IEEE_QUIET_GT_R)
  end interface ieee_quiet_gt
  PRIVATE_R(IEEE_QUIET_GT)
#undef IEEE_QUIET_GT_R

#define IEEE_QUIET_LE_R(AKIND) \
  elemental logical function ieee_quiet_le_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_quiet_le_a##AKIND;
  interface ieee_quiet_le
    SPECIFICS_R(IEEE_QUIET_LE_R)
  end interface ieee_quiet_le
  PRIVATE_R(IEEE_QUIET_LE)
#undef IEEE_QUIET_LE_R

#define IEEE_QUIET_LT_R(AKIND) \
  elemental logical function ieee_quiet_lt_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_quiet_lt_a##AKIND;
  interface ieee_quiet_lt
    SPECIFICS_R(IEEE_QUIET_LT_R)
  end interface ieee_quiet_lt
  PRIVATE_R(IEEE_QUIET_LT)
#undef IEEE_QUIET_LT_R

#define IEEE_QUIET_NE_R(AKIND) \
  elemental logical function ieee_quiet_ne_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_quiet_ne_a##AKIND;
  interface ieee_quiet_ne
    SPECIFICS_R(IEEE_QUIET_NE_R)
  end interface ieee_quiet_ne
  PRIVATE_R(IEEE_QUIET_NE)
#undef IEEE_QUIET_NE_R

! When kind argument is present, kind(result) is value(kind), not kind(kind).
! That is not known here, so return real(16).
#define IEEE_REAL_I(AKIND) \
  elemental real function ieee_real_i##AKIND(a); \
    integer(AKIND), intent(in) :: a; \
  end function ieee_real_i##AKIND;
#define IEEE_REAL_R(AKIND) \
  elemental real function ieee_real_a##AKIND(a); \
    real(AKIND), intent(in) :: a; \
  end function ieee_real_a##AKIND;
#define IEEE_REAL_II(AKIND, KKIND) \
  elemental real(16) function ieee_real_i##AKIND##_i##KKIND(a, kind); \
    integer(AKIND), intent(in) :: a; \
    integer(KKIND), intent(in) :: kind; \
  end function ieee_real_i##AKIND##_i##KKIND;
#define IEEE_REAL_RI(AKIND, KKIND) \
  elemental real(16) function ieee_real_a##AKIND##_i##KKIND(a, kind); \
    real(AKIND), intent(in) :: a; \
    integer(KKIND), intent(in) :: kind; \
  end function ieee_real_a##AKIND##_i##KKIND;
  interface ieee_real
    SPECIFICS_I(IEEE_REAL_I)
    SPECIFICS_R(IEEE_REAL_R)
    SPECIFICS_II(IEEE_REAL_II)
    SPECIFICS_RI(IEEE_REAL_RI)
  end interface ieee_real
  PRIVATE_I(IEEE_REAL)
  PRIVATE_R(IEEE_REAL)
  PRIVATE_II(IEEE_REAL)
  PRIVATE_RI(IEEE_REAL)
#undef IEEE_REAL_I
#undef IEEE_REAL_R
#undef IEEE_REAL_II
#undef IEEE_REAL_RI

#define IEEE_REM_RR(XKIND, YKIND) \
  elemental real(XKIND) function ieee_rem_a##XKIND##_a##YKIND(x, y); \
    real(XKIND), intent(in) :: x; \
    real(YKIND), intent(in) :: y; \
  end function ieee_rem_a##XKIND##_a##YKIND;
  interface ieee_rem
    SPECIFICS_RR(IEEE_REM_RR)
  end interface ieee_rem
  PRIVATE_RR(IEEE_REM)
#undef IEEE_REM_RR

#define IEEE_RINT_R(XKIND) \
  elemental real(XKIND) function ieee_rint_a##XKIND(x, round); \
    import ieee_round_type; \
    real(XKIND), intent(in) :: x; \
    type(ieee_round_type), optional, intent(in) :: round; \
  end function ieee_rint_a##XKIND;
  interface ieee_rint
    SPECIFICS_R(IEEE_RINT_R)
  end interface ieee_rint
  PRIVATE_R(IEEE_RINT)
#undef IEEE_RINT_R

#define IEEE_SET_ROUNDING_MODE_I(RKIND) \
  subroutine ieee_set_rounding_mode_i##RKIND(round_value, radix); \
    import ieee_round_type; \
    type(ieee_round_type), intent(in) :: round_value; \
    integer(RKIND), intent(in) :: radix; \
  end subroutine ieee_set_rounding_mode_i##RKIND;
  interface ieee_set_rounding_mode
    subroutine ieee_set_rounding_mode(round_value)
      import ieee_round_type
      type(ieee_round_type), intent(in) :: round_value
    end subroutine ieee_set_rounding_mode
    SPECIFICS_I(IEEE_SET_ROUNDING_MODE_I)
  end interface ieee_set_rounding_mode
  PRIVATE_I(IEEE_SET_ROUNDING_MODE)
#undef IEEE_SET_ROUNDING_MODE_I

#define IEEE_SET_UNDERFLOW_MODE_L(GKIND) \
  subroutine ieee_set_underflow_mode_l##GKIND(gradual); \
    logical(GKIND), intent(in) :: gradual; \
  end subroutine ieee_set_underflow_mode_l##GKIND;
  interface ieee_set_underflow_mode
    SPECIFICS_L(IEEE_SET_UNDERFLOW_MODE_L)
  end interface ieee_set_underflow_mode
  PRIVATE_L(IEEE_SET_UNDERFLOW_MODE)
#undef IEEE_SET_UNDERFLOW_MODE_L

#define IEEE_SIGNALING_EQ_R(AKIND) \
  elemental logical function ieee_signaling_eq_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_signaling_eq_a##AKIND;
  interface ieee_signaling_eq
    SPECIFICS_R(IEEE_SIGNALING_EQ_R)
  end interface ieee_signaling_eq
  PRIVATE_R(IEEE_SIGNALING_EQ)
#undef IEEE_SIGNALING_EQ_R

#define IEEE_SIGNALING_GE_R(AKIND) \
  elemental logical function ieee_signaling_ge_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_signaling_ge_a##AKIND;
  interface ieee_signaling_ge
    SPECIFICS_R(IEEE_SIGNALING_GE_R)
  end interface ieee_signaling_ge
  PRIVATE_R(IEEE_SIGNALING_GE)
#undef IEEE_SIGNALING_GE_R

#define IEEE_SIGNALING_GT_R(AKIND) \
  elemental logical function ieee_signaling_gt_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_signaling_gt_a##AKIND;
  interface ieee_signaling_gt
    SPECIFICS_R(IEEE_SIGNALING_GT_R)
  end interface ieee_signaling_gt
  PRIVATE_R(IEEE_SIGNALING_GT)
#undef IEEE_SIGNALING_GT_R

#define IEEE_SIGNALING_LE_R(AKIND) \
  elemental logical function ieee_signaling_le_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_signaling_le_a##AKIND;
  interface ieee_signaling_le
    SPECIFICS_R(IEEE_SIGNALING_LE_R)
  end interface ieee_signaling_le
  PRIVATE_R(IEEE_SIGNALING_LE)
#undef IEEE_SIGNALING_LE_R

#define IEEE_SIGNALING_LT_R(AKIND) \
  elemental logical function ieee_signaling_lt_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_signaling_lt_a##AKIND;
  interface ieee_signaling_lt
    SPECIFICS_R(IEEE_SIGNALING_LT_R)
  end interface ieee_signaling_lt
  PRIVATE_R(IEEE_SIGNALING_LT)
#undef IEEE_SIGNALING_LT_R

#define IEEE_SIGNALING_NE_R(AKIND) \
  elemental logical function ieee_signaling_ne_a##AKIND(a, b); \
    real(AKIND), intent(in) :: a, b; \
  end function ieee_signaling_ne_a##AKIND;
  interface ieee_signaling_ne
    SPECIFICS_R(IEEE_SIGNALING_NE_R)
  end interface ieee_signaling_ne
  PRIVATE_R(IEEE_SIGNALING_NE)
#undef IEEE_SIGNALING_NE_R

#define IEEE_SIGNBIT_R(XKIND) \
  elemental logical function ieee_signbit_a##XKIND(x); \
    real(XKIND), intent(in) :: x; \
  end function ieee_signbit_a##XKIND;
  interface ieee_signbit
    SPECIFICS_R(IEEE_SIGNBIT_R)
  end interface ieee_signbit
  PRIVATE_R(IEEE_SIGNBIT)
#undef IEEE_SIGNBIT_R

#define IEEE_SUPPORT_ROUNDING_R(XKIND) \
  pure logical function ieee_support_rounding_a##XKIND(round_value, x); \
    import ieee_round_type; \
    type(ieee_round_type), intent(in) :: round_value; \
    real(XKIND), intent(in) :: x(..); \
  end function ieee_support_rounding_a##XKIND;
  interface ieee_support_rounding
    pure logical function ieee_support_rounding(round_value)
      import ieee_round_type
      type(ieee_round_type), intent(in) :: round_value
    end function ieee_support_rounding
    SPECIFICS_R(IEEE_SUPPORT_ROUNDING_R)
  end interface ieee_support_rounding
  PRIVATE_R(IEEE_SUPPORT_ROUNDING)
#undef IEEE_SUPPORT_ROUNDING_R

#define IEEE_UNORDERED_RR(XKIND, YKIND) \
 elemental logical function ieee_unordered_a##XKIND##_a##YKIND(x, y); \
    real(XKIND), intent(in) :: x; \
    real(YKIND), intent(in) :: y; \
  end function ieee_unordered_a##XKIND##_a##YKIND;
  interface ieee_unordered
    SPECIFICS_RR(IEEE_UNORDERED_RR)
  end interface ieee_unordered
  PRIVATE_RR(IEEE_UNORDERED)
#undef IEEE_UNORDERED_RR

#define IEEE_VALUE_R(XKIND) \
  elemental real(XKIND) function ieee_value_a##XKIND(x, class); \
    import ieee_class_type; \
    real(XKIND), intent(in) :: x; \
    type(ieee_class_type), intent(in) :: class; \
  end function ieee_value_a##XKIND;
  interface ieee_value
    SPECIFICS_R(IEEE_VALUE_R)
  end interface ieee_value
  PRIVATE_R(IEEE_VALUE)
#undef IEEE_VALUE_R

end module ieee_arithmetic
