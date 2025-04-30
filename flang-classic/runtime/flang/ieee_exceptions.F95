! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

#ifndef DESC_I8
#define IEEE_EXCEPTIONS ieee_exceptions
#else
#define IEEE_EXCEPTIONS ieee_exceptions_la
#endif

module IEEE_EXCEPTIONS
#ifdef PGDLL
!DEC$ ATTRIBUTES DLLEXPORT :: IEEE_EXCEPTIONS
#endif

  ! The spec section 14.9 says these are supported in ieee_exceptions
  ! inquiry
  !   ieee_support_flag
  !   ieee_support_halting
  ! elemental subroutines
  !   ieee_get_flag
  !   ieee_get_halting_mode
  ! nonelemental subroutines
  !   ieee_get_status
  !   ieee_set_flag
  !   ieee_set_halting_mode
  !   ieee_set_status
  !
  ! These are from fenv.h on linux64
#ifdef TARGET_LINUX_ARM
  integer, private, parameter :: FE_INVALID   = 1
  integer, private, parameter :: FE_DIVBYZERO = 2
  integer, private, parameter :: FE_OVERFLOW  = 4
  integer, private, parameter :: FE_UNDERFLOW = 8
  integer, private, parameter :: FE_INEXACT   = 16
  integer, private, parameter :: FE_DENORM    = 0
  ! FE_DENORM is not currently supported on arm 
#elif defined(TARGET_LINUX_POWER)
  ! FE_DENORM is not supported on Power
  integer, private, parameter :: FE_INVALID   = X'20000000'
  integer, private, parameter :: FE_DIVBYZERO = X'04000000'
  integer, private, parameter :: FE_OVERFLOW  = X'10000000'
  integer, private, parameter :: FE_UNDERFLOW = X'08000000'
  integer, private, parameter :: FE_INEXACT   = X'02000000'
  integer, private, parameter :: FE_DENORM    = 0
#else
  integer, private, parameter :: FE_INVALID   = 1
#if defined(PGFLANG)
  ! FE_DENORM is not supported in Flang
  integer, private, parameter :: FE_DENORM    = 0
#else
  integer, private, parameter :: FE_DENORM    = 2
#endif
  integer, private, parameter :: FE_DIVBYZERO = 4
  integer, private, parameter :: FE_OVERFLOW  = 8
  integer, private, parameter :: FE_UNDERFLOW = 16
  integer, private, parameter :: FE_INEXACT   = 32
#endif

! Derived types and parameters
  type ieee_flag_type
    private
    integer :: ft
  end type ieee_flag_type

  type ieee_status_type
    private
    integer :: mxcsr
    integer :: x87cw
    integer :: x87sw
  end type ieee_status_type

  type(ieee_flag_type), parameter :: ieee_invalid        = &
     ieee_flag_type(FE_INVALID)

  type(ieee_flag_type), parameter :: ieee_denorm         = &
     ieee_flag_type(FE_DENORM)

  type(ieee_flag_type), parameter :: ieee_divide_by_zero = &
     ieee_flag_type(FE_DIVBYZERO)

  type(ieee_flag_type), parameter :: ieee_overflow       = &
     ieee_flag_type(FE_OVERFLOW)

  type(ieee_flag_type), parameter :: ieee_underflow      = &
     ieee_flag_type(FE_UNDERFLOW)

  type(ieee_flag_type), parameter :: ieee_inexact        = &
     ieee_flag_type(FE_INEXACT)

  type(ieee_flag_type), parameter :: ieee_usual(3) = &
            (/ieee_overflow, ieee_divide_by_zero, ieee_invalid/)

  type(ieee_flag_type), parameter :: ieee_all(5) = &
            (/ieee_usual, ieee_underflow, ieee_inexact/)

  ! Generic interfaces
  !--------------------
  interface ieee_support_flag
    module procedure ieee_support_flagnox, ieee_support_flagr
  end interface

  interface ieee_set_flag
    module procedure ieee_set_flag_scalar
    module procedure ieee_set_flag_array
    module procedure ieee_set_flag_arrscal
    module procedure ieee_set_flag_scalar_l8
    module procedure ieee_set_flag_array_l8
    module procedure ieee_set_flag_arrscal_l8
  end interface

  interface ieee_set_halting_mode
    module procedure ieee_set_halting_mode_scalar
    module procedure ieee_set_halting_mode_array
    module procedure ieee_set_halting_mode_arrscal
    module procedure ieee_set_halting_mode_scalar_l8
    module procedure ieee_set_halting_mode_array_l8
    module procedure ieee_set_halting_mode_arrscal_l8
  end interface

  interface ieee_get_flag
    module procedure ieee_get_flag
    module procedure ieee_get_flag_l8
  end interface

  interface ieee_get_halting_mode
    module procedure ieee_get_halting_mode
    module procedure ieee_get_halting_mode_l8
  end interface

  ! External interfaces
  !--------------------
  interface
    pure integer function __fenv_fegetexceptflag(flagp, exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int)        :: flagp
      integer(c_int), value :: exc
    end function __fenv_fegetexceptflag

    pure integer function __fenv_fesetexceptflag(flagp, exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int)        :: flagp
      integer(c_int), value :: exc
    end function __fenv_fesetexceptflag

    pure integer function __fenv_fetestexcept(exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: exc
    end function __fenv_fetestexcept

    pure integer function __fenv_feclearexcept(exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: exc
    end function __fenv_feclearexcept

    pure integer function __fenv_feraiseexcept(exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: exc
    end function __fenv_feraiseexcept

    pure integer function __fenv_feenableexcept(exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: exc
    end function __fenv_feenableexcept

    pure integer function __fenv_fedisableexcept(exc) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: exc
    end function __fenv_fedisableexcept

    pure integer function __fenv_fegetexcept() bind(c)
    end function __fenv_fegetexcept

    integer function __fenv_fegetenv(env) bind(c)
      import
      type(ieee_status_type) :: env
    end function __fenv_fegetenv

    integer function __fenv_fesetenv(env) bind(c)
      import
      type(ieee_status_type) :: env
    end function __fenv_fesetenv
  end interface

!--------------------------------------------------------------------------
! Inquiry functions for IEEE exceptions
contains
  logical function ieee_support_flagnox(flag)
    type(ieee_flag_type)  :: flag
    ieee_support_flagnox = .true.
    return
  end function ieee_support_flagnox

  logical function ieee_support_flagr(flag, x)
!dir$ ignore_tkr (kr) x
    type(ieee_flag_type)  :: flag
    real :: x
    ieee_support_flagr = .true.
    return
  end function ieee_support_flagr

  logical function ieee_support_halting(flag)
    type(ieee_flag_type) :: flag
    ieee_support_halting = .true.
    return
  end function ieee_support_halting

!--------------------------------------------------------------------------
  elemental subroutine ieee_get_flag(flag, flag_value)
    type(ieee_flag_type)  :: flag
    logical, intent(out) :: flag_value
    flag_value = (__fenv_fetestexcept(flag%ft) .eq. flag%ft)
    return
  end subroutine

  elemental subroutine ieee_get_flag_l8(flag, flag_value)
    type(ieee_flag_type)  :: flag
    logical*8, intent(out) :: flag_value
    flag_value = (__fenv_fetestexcept(flag%ft) .eq. flag%ft)
    return
  end subroutine

  elemental subroutine ieee_get_halting_mode(flag, halting)
    type(ieee_flag_type)  :: flag
    logical, intent(out) :: halting
    halting = (iand(__fenv_fegetexcept(),flag%ft) .ne. 0)
    return
  end subroutine

  elemental subroutine ieee_get_halting_mode_l8(flag, halting)
    type(ieee_flag_type)  :: flag
    logical*8, intent(out) :: halting
    halting = (iand(__fenv_fegetexcept(),flag%ft) .ne. 0)
    return
  end subroutine

!--------------------------------------------------------------------------
  pure subroutine ieee_set_flag_scalar(flag, flag_value)
    type(ieee_flag_type) :: flag
    logical, intent(in)  :: flag_value
    if (flag_value) then
      i = __fenv_feraiseexcept(flag%ft)
    else
      i = __fenv_feclearexcept(flag%ft)
    endif
  end subroutine ieee_set_flag_scalar

  pure subroutine ieee_set_flag_array(flag, flag_value)
    type(ieee_flag_type), dimension(:) :: flag
    logical, intent(in), dimension(:)  :: flag_value
    integer flagp, flagv
    flagp = 0
    flagv = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (flag_value(i)) then
        flagp = ior(flagp,flag(i)%ft)
      endif
      flagv = ior(flagv,flag(i)%ft)
    end do
    i = __fenv_fesetexceptflag(flagp, flagv)
  end subroutine ieee_set_flag_array
  
  pure subroutine ieee_set_flag_arrscal(flag, flag_value)
    type(ieee_flag_type), dimension(:) :: flag
    logical, intent(in) :: flag_value
    integer flagp, flagv
    flagp = 0
    flagv = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (flag_value) then
        flagp = ior(flagp,flag(i)%ft)
      endif
      flagv = ior(flagv,flag(i)%ft)
    end do
    i = __fenv_fesetexceptflag(flagp, flagv)
  end subroutine ieee_set_flag_arrscal

  subroutine ieee_set_halting_mode_scalar(flag, halting)
    type(ieee_flag_type) :: flag
    logical, intent(in)  :: halting
    if (halting) then
      i = __fenv_feenableexcept(flag%ft)
    else
      i = __fenv_fedisableexcept(flag%ft)
    endif
  end subroutine ieee_set_halting_mode_scalar

  subroutine ieee_set_halting_mode_array(flag, halting)
    type(ieee_flag_type), dimension(:) :: flag
    logical, intent(in), dimension(:)  :: halting
    integer flagp, flagn
    flagp = 0
    flagn = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (halting(i)) then
        flagp = ior(flagp,flag(i)%ft)
      else
        flagn = ior(flagn,flag(i)%ft)
      endif
    end do
    i = __fenv_feenableexcept(flagp)
    i = __fenv_fedisableexcept(flagn)
  end subroutine ieee_set_halting_mode_array

  subroutine ieee_set_halting_mode_arrscal(flag, halting)
    type(ieee_flag_type), dimension(:) :: flag
    logical, intent(in) :: halting
    integer flagp, flagn
    flagp = 0
    flagn = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (halting) then
        flagp = ior(flagp,flag(i)%ft)
      else
        flagn = ior(flagn,flag(i)%ft)
      endif
    end do
    i = __fenv_feenableexcept(flagp)
    i = __fenv_fedisableexcept(flagn)
  end subroutine ieee_set_halting_mode_arrscal

!--------------------------------------------------------------------------
  pure subroutine ieee_set_flag_scalar_l8(flag, flag_value)
    type(ieee_flag_type) :: flag
    logical*8, intent(in)  :: flag_value
    if (flag_value) then
      i = __fenv_feraiseexcept(flag%ft)
    else
      i = __fenv_feclearexcept(flag%ft)
    endif
  end subroutine ieee_set_flag_scalar_l8

  pure subroutine ieee_set_flag_array_l8(flag, flag_value)
    type(ieee_flag_type), dimension(:) :: flag
    logical*8, intent(in), dimension(:)  :: flag_value
    integer flagp, flagv
    flagp = 0
    flagv = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (flag_value(i)) then
        flagp = ior(flagp,flag(i)%ft)
      endif
      flagv = ior(flagv,flag(i)%ft)
    end do
    i = __fenv_fesetexceptflag(flagp, flagv)
  end subroutine ieee_set_flag_array_l8
  
  pure subroutine ieee_set_flag_arrscal_l8(flag, flag_value)
    type(ieee_flag_type), dimension(:) :: flag
    logical*8, intent(in) :: flag_value
    integer flagp, flagv
    flagp = 0
    flagv = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (flag_value) then
        flagp = ior(flagp,flag(i)%ft)
      endif
      flagv = ior(flagv,flag(i)%ft)
    end do
    i = __fenv_fesetexceptflag(flagp, flagv)
  end subroutine ieee_set_flag_arrscal_l8

  subroutine ieee_set_halting_mode_scalar_l8(flag, halting)
    type(ieee_flag_type) :: flag
    logical*8, intent(in)  :: halting
    if (halting) then
      i = __fenv_feenableexcept(flag%ft)
    else
      i = __fenv_fedisableexcept(flag%ft)
    endif
  end subroutine ieee_set_halting_mode_scalar_l8

  subroutine ieee_set_halting_mode_array_l8(flag, halting)
    type(ieee_flag_type), dimension(:) :: flag
    logical*8, intent(in), dimension(:)  :: halting
    integer flagp, flagn
    flagp = 0
    flagn = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (halting(i)) then
        flagp = ior(flagp,flag(i)%ft)
      else
        flagn = ior(flagn,flag(i)%ft)
      endif
    end do
    i = __fenv_feenableexcept(flagp)
    i = __fenv_fedisableexcept(flagn)
  end subroutine ieee_set_halting_mode_array_l8

  subroutine ieee_set_halting_mode_arrscal_l8(flag, halting)
    type(ieee_flag_type), dimension(:) :: flag
    logical*8, intent(in) :: halting
    integer flagp, flagn
    flagp = 0
    flagn = 0
    do i = lbound(flag,1),ubound(flag,1)
      if (halting) then
        flagp = ior(flagp,flag(i)%ft)
      else
        flagn = ior(flagn,flag(i)%ft)
      endif
    end do
    i = __fenv_feenableexcept(flagp)
    i = __fenv_fedisableexcept(flagn)
  end subroutine ieee_set_halting_mode_arrscal_l8

!--------------------------------------------------------------------------
  subroutine ieee_get_status(status_value)
    type(ieee_status_type), intent(out) :: status_value
    i = __fenv_fegetenv(status_value)
    return
  end subroutine

  subroutine ieee_set_status(status_value)
    type(ieee_status_type), intent(in) :: status_value
    i = __fenv_fesetenv(status_value)
    return
  end subroutine

end module IEEE_EXCEPTIONS
