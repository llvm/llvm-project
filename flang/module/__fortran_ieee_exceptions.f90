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

#include '../include/flang/Runtime/magic-numbers.h'

module __fortran_ieee_exceptions
  use __fortran_builtins, only: &
    ieee_flag_type => __builtin_ieee_flag_type, &
    ieee_support_flag => __builtin_ieee_support_flag, &
    ieee_support_halting => __builtin_ieee_support_halting, &
    ieee_invalid => __builtin_ieee_invalid, &
    ieee_overflow => __builtin_ieee_overflow, &
    ieee_divide_by_zero => __builtin_ieee_divide_by_zero, &
    ieee_underflow => __builtin_ieee_underflow, &
    ieee_inexact => __builtin_ieee_inexact, &
    ieee_denorm => __builtin_ieee_denorm
  implicit none
  private

  public :: ieee_flag_type, ieee_support_flag, ieee_support_halting
  public :: ieee_invalid, ieee_overflow, ieee_divide_by_zero, ieee_underflow, &
            ieee_inexact, ieee_denorm

  type(ieee_flag_type), parameter, public :: &
    ieee_usual(*) = [ ieee_overflow, ieee_divide_by_zero, ieee_invalid ], &
    ieee_all(*) = [ ieee_usual, ieee_underflow, ieee_inexact ]

  type, public :: ieee_modes_type ! Fortran 2018, 17.7
    private ! opaque fenv.h femode_t data; code will access only one component
    integer(kind=4) :: __data(_FORTRAN_RUNTIME_IEEE_FEMODE_T_EXTENT)
    integer(kind=1), allocatable :: __allocatable_data(:)
  end type ieee_modes_type

  type, public :: ieee_status_type ! Fortran 2018, 17.7
    private ! opaque fenv.h fenv_t data; code will access only one component
    integer(kind=4) :: __data(_FORTRAN_RUNTIME_IEEE_FENV_T_EXTENT)
    integer(kind=1), allocatable :: __allocatable_data(:)
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

#define IEEE_GET_FLAG_L(FVKIND) \
  elemental subroutine ieee_get_flag_l##FVKIND(flag, flag_value); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    logical(FVKIND), intent(out) :: flag_value; \
  end subroutine ieee_get_flag_l##FVKIND;
  interface ieee_get_flag
    SPECIFICS_L(IEEE_GET_FLAG_L)
  end interface ieee_get_flag
  public :: ieee_get_flag
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
  public :: ieee_get_halting_mode
#undef IEEE_GET_HALTING_MODE_L

  interface ieee_get_modes
    pure subroutine ieee_get_modes_0(modes)
      import ieee_modes_type
      type(ieee_modes_type), intent(out) :: modes
    end subroutine ieee_get_modes_0
  end interface
  public :: ieee_get_modes

  interface ieee_get_status
    pure subroutine ieee_get_status_0(status)
      import ieee_status_type
      type(ieee_status_type), intent(out) :: status
    end subroutine ieee_get_status_0
  end interface
  public :: ieee_get_status

#define IEEE_SET_FLAG_L(FVKIND) \
  elemental subroutine ieee_set_flag_l##FVKIND(flag, flag_value); \
    import ieee_flag_type; \
    type(ieee_flag_type), intent(in) :: flag; \
    logical(FVKIND), intent(in) :: flag_value; \
  end subroutine ieee_set_flag_l##FVKIND;
  interface ieee_set_flag
    SPECIFICS_L(IEEE_SET_FLAG_L)
  end interface ieee_set_flag
  public :: ieee_set_flag
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
  public :: ieee_set_halting_mode
#undef IEEE_SET_HALTING_MODE_L

  interface ieee_set_modes
    subroutine ieee_set_modes_0(modes)
      import ieee_modes_type
      type(ieee_modes_type), intent(in) :: modes
    end subroutine ieee_set_modes_0
  end interface
  public :: ieee_set_modes

  interface ieee_set_status
    pure subroutine ieee_set_status_0(status)
      import ieee_status_type
      type(ieee_status_type), intent(in) :: status
    end subroutine ieee_set_status_0
  end interface
  public :: ieee_set_status

end module __fortran_ieee_exceptions
