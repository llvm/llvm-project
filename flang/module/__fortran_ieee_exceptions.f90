!===-- module/__fortran_ieee_exceptions.f90 --------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2018, clause 17
! The content of the standard intrinsic IEEE_EXCEPTIONS module is packaged
! here under another name so that IEEE_ARITHMETIC can USE it and export its
! declarations without clashing with a non-intrinsic module in a program.

include '../include/flang/Runtime/magic-numbers.h'

module __fortran_ieee_exceptions

  type :: ieee_flag_type ! Fortran 2018, 17.2 & 17.3
    private
    integer(kind=1) :: flag = 0
  end type ieee_flag_type

  type(ieee_flag_type), parameter :: &
    ieee_invalid = ieee_flag_type(_FORTRAN_RUNTIME_IEEE_INVALID), &
    ieee_overflow = ieee_flag_type(_FORTRAN_RUNTIME_IEEE_OVERFLOW), &
    ieee_divide_by_zero = &
        ieee_flag_type(_FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO), &
    ieee_underflow = ieee_flag_type(_FORTRAN_RUNTIME_IEEE_UNDERFLOW), &
    ieee_inexact = ieee_flag_type(_FORTRAN_RUNTIME_IEEE_INEXACT), &
    ieee_denorm = ieee_flag_type(_FORTRAN_RUNTIME_IEEE_DENORM) ! extension

  type(ieee_flag_type), parameter :: &
    ieee_usual(*) = [ ieee_overflow, ieee_divide_by_zero, ieee_invalid ], &
    ieee_all(*) = [ ieee_usual, ieee_underflow, ieee_inexact ]

  type :: ieee_modes_type ! Fortran 2018, 17.7
    private ! opaque fenv.h femode_t data
    integer(kind=4) :: __data(_FORTRAN_RUNTIME_IEEE_FEMODE_T_EXTENT)
  end type ieee_modes_type

  type :: ieee_status_type ! Fortran 2018, 17.7
    private ! opaque fenv.h fenv_t data
    integer(kind=4) :: __data(_FORTRAN_RUNTIME_IEEE_FENV_T_EXTENT)
  end type ieee_status_type

! Define specifics with 1 LOGICAL or REAL argument for generic G.
#define SPECIFICS_L(G) \
  G(1) G(2) G(4) G(8)
#if __x86_64__
#define SPECIFICS_R(G) \
  G(2) G(3) G(4) G(8) G(10) G(16)
#else
#define SPECIFICS_R(G) \
  G(2) G(3) G(4) G(8) G(16)
#endif

! Set PRIVATE accessibility for specifics with 1 LOGICAL or REAL argument for
! generic G.
#define PRIVATE_L(G) private :: \
  G##_l1, G##_l2, G##_l4, G##_l8
#if __x86_64__
#define PRIVATE_R(G) private :: \
  G##_a2, G##_a3, G##_a4, G##_a8, G##_a10, G##_a16
#else
#define PRIVATE_R(G) private :: \
  G##_a2, G##_a3, G##_a4, G##_a8, G##_a16
#endif

#define IEEE_GET_FLAG_L(FVKIND) \
  elemental subroutine ieee_get_flag_l##FVKIND(flag, flag_value); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    logical(FVKIND), intent(out) :: flag_value; \
  end subroutine ieee_get_flag_l##FVKIND;
  interface ieee_get_flag
    SPECIFICS_L(IEEE_GET_FLAG_L)
  end interface ieee_get_flag
  PRIVATE_L(IEEE_GET_FLAG)
#undef IEEE_GET_FLAG_L

#define IEEE_GET_HALTING_MODE_L(HKIND) \
  elemental subroutine ieee_get_halting_mode_l##HKIND(flag, halting); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    logical(HKIND), intent(out) :: halting; \
  end subroutine ieee_get_halting_mode_l##HKIND;
  interface ieee_get_halting_mode
    SPECIFICS_L(IEEE_GET_HALTING_MODE_L)
  end interface ieee_get_halting_mode
  PRIVATE_L(IEEE_GET_HALTING_MODE)
#undef IEEE_GET_HALTING_MODE_L

  interface ieee_get_modes
    pure subroutine ieee_get_modes_0(modes)
      import ieee_modes_type
      type(ieee_modes_type), intent(out) :: modes
    end subroutine ieee_get_modes_0
  end interface

  interface ieee_get_status
    pure subroutine ieee_get_status_0(status)
      import ieee_status_type
      type(ieee_status_type), intent(out) :: status
    end subroutine ieee_get_status_0
  end interface

#define IEEE_SET_FLAG_L(FVKIND) \
  elemental subroutine ieee_set_flag_l##FVKIND(flag, flag_value); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    logical(FVKIND), intent(in) :: flag_value; \
  end subroutine ieee_set_flag_l##FVKIND;
  interface ieee_set_flag
    SPECIFICS_L(IEEE_SET_FLAG_L)
  end interface ieee_set_flag
  PRIVATE_L(IEEE_SET_FLAG)
#undef IEEE_SET_FLAG_L

#define IEEE_SET_HALTING_MODE_L(HKIND) \
  elemental subroutine ieee_set_halting_mode_l##HKIND(flag, halting); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    logical(HKIND), intent(in) :: halting; \
  end subroutine ieee_set_halting_mode_l##HKIND;
  interface ieee_set_halting_mode
    SPECIFICS_L(IEEE_SET_HALTING_MODE_L)
  end interface ieee_set_halting_mode
  PRIVATE_L(IEEE_SET_HALTING_MODE)
#undef IEEE_SET_HALTING_MODE_L

  interface ieee_set_modes
    subroutine ieee_set_modes_0(modes)
      import ieee_modes_type
      type(ieee_modes_type), intent(in) :: modes
    end subroutine ieee_set_modes_0
  end interface

  interface ieee_set_status
    subroutine ieee_set_status_0(status)
      import ieee_status_type
      type(ieee_status_type), intent(in) :: status
    end subroutine ieee_set_status_0
  end interface

#define IEEE_SUPPORT_FLAG_R(XKIND) \
  pure logical function ieee_support_flag_a##XKIND(flag, x); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    real(XKIND), intent(in) :: x(..); \
  end function ieee_support_flag_a##XKIND;
  interface ieee_support_flag
    pure logical function ieee_support_flag_0(flag)
      import ieee_flag_type
      type(ieee_flag_type), intent(in) :: flag
    end function ieee_support_flag_0
    SPECIFICS_R(IEEE_SUPPORT_FLAG_R)
  end interface ieee_support_flag
  PRIVATE_R(IEEE_SUPPORT_FLAG)
#undef IEEE_SUPPORT_FLAG_R

  interface ieee_support_halting
    pure logical function ieee_support_halting_0(flag)
      import ieee_flag_type
      type(ieee_flag_type), intent(in) :: flag
    end function ieee_support_halting_0
  end interface

end module __fortran_ieee_exceptions
