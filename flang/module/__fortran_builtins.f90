!===-- module/__fortran_builtins.f90 ---------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

include '../include/flang/Runtime/magic-numbers.h'

! These naming shenanigans prevent names from Fortran intrinsic modules
! from being usable on INTRINSIC statements, and force the program
! to USE the standard intrinsic modules in order to access the
! standard names of the procedures.
module __fortran_builtins
  implicit none

  ! Set PRIVATE by default to explicitly only export what is meant
  ! to be exported by this MODULE.
  private

  intrinsic :: __builtin_c_loc
  public :: __builtin_c_loc

  intrinsic :: __builtin_c_f_pointer
  public :: __builtin_c_f_pointer

  intrinsic :: sizeof ! extension
  public :: sizeof

  intrinsic :: selected_int_kind
  integer, parameter :: int64 = selected_int_kind(18)

  type, bind(c), public :: __builtin_c_ptr
    integer(kind=int64), private :: __address
  end type

  type, bind(c), public :: __builtin_c_funptr
    integer(kind=int64), private :: __address
  end type

  type, public :: __builtin_event_type
    integer(kind=int64), private :: __count
  end type

  type, public :: __builtin_notify_type
    integer(kind=int64), private :: __count
  end type

  type, public :: __builtin_lock_type
    integer(kind=int64), private :: __count
  end type

  type, public :: __builtin_ieee_flag_type
    integer(kind=1), private :: flag = 0
  end type

  type(__builtin_ieee_flag_type), parameter, public :: &
    __builtin_ieee_invalid = &
      __builtin_ieee_flag_type(_FORTRAN_RUNTIME_IEEE_INVALID), &
    __builtin_ieee_overflow = &
      __builtin_ieee_flag_type(_FORTRAN_RUNTIME_IEEE_OVERFLOW), &
    __builtin_ieee_divide_by_zero = &
      __builtin_ieee_flag_type(_FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO), &
    __builtin_ieee_underflow = &
      __builtin_ieee_flag_type(_FORTRAN_RUNTIME_IEEE_UNDERFLOW), &
    __builtin_ieee_inexact = &
      __builtin_ieee_flag_type(_FORTRAN_RUNTIME_IEEE_INEXACT), &
    __builtin_ieee_denorm = & ! extension
      __builtin_ieee_flag_type(_FORTRAN_RUNTIME_IEEE_DENORM)

  type, public :: __builtin_ieee_round_type
    integer(kind=1), private :: mode = 0
  end type

  type(__builtin_ieee_round_type), parameter, public :: &
    __builtin_ieee_to_zero = &
      __builtin_ieee_round_type(_FORTRAN_RUNTIME_IEEE_TO_ZERO), &
    __builtin_ieee_nearest = &
      __builtin_ieee_round_type(_FORTRAN_RUNTIME_IEEE_NEAREST), &
    __builtin_ieee_up = &
      __builtin_ieee_round_type(_FORTRAN_RUNTIME_IEEE_UP), &
    __builtin_ieee_down = &
      __builtin_ieee_round_type(_FORTRAN_RUNTIME_IEEE_DOWN), &
    __builtin_ieee_away = &
      __builtin_ieee_round_type(_FORTRAN_RUNTIME_IEEE_AWAY), &
    __builtin_ieee_other = &
      __builtin_ieee_round_type(_FORTRAN_RUNTIME_IEEE_OTHER)

  type, public :: __builtin_team_type
    integer(kind=int64), private :: __id
  end type

  integer, parameter, public :: __builtin_atomic_int_kind = selected_int_kind(18)
  integer, parameter, public :: &
    __builtin_atomic_logical_kind = __builtin_atomic_int_kind

  type, public :: __builtin_dim3
    integer :: x=1, y=1, z=1
  end type
  type(__builtin_dim3), public :: &
    __builtin_threadIdx, __builtin_blockDim, __builtin_blockIdx, &
    __builtin_gridDim
  integer, parameter, public :: __builtin_warpsize = 32

  intrinsic :: __builtin_fma
  intrinsic :: __builtin_ieee_is_nan, __builtin_ieee_is_negative, &
    __builtin_ieee_is_normal
  intrinsic :: __builtin_ieee_next_after, __builtin_ieee_next_down, &
    __builtin_ieee_next_up
  intrinsic :: scale ! for ieee_scalb
  intrinsic :: __builtin_ieee_selected_real_kind
  intrinsic :: __builtin_ieee_support_datatype, &
    __builtin_ieee_support_denormal, __builtin_ieee_support_divide, &
    __builtin_ieee_support_flag, __builtin_ieee_support_halting, &
    __builtin_ieee_support_inf, __builtin_ieee_support_io, &
    __builtin_ieee_support_nan, __builtin_ieee_support_rounding, &
    __builtin_ieee_support_sqrt, &
    __builtin_ieee_support_standard, __builtin_ieee_support_subnormal, &
    __builtin_ieee_support_underflow_control
  public :: __builtin_fma
  public :: __builtin_ieee_is_nan, __builtin_ieee_is_negative, &
    __builtin_ieee_is_normal
  public :: __builtin_ieee_next_after, __builtin_ieee_next_down, &
    __builtin_ieee_next_up
  public :: scale ! for ieee_scalb
  public :: __builtin_ieee_selected_real_kind
  public :: __builtin_ieee_support_datatype, &
    __builtin_ieee_support_denormal, __builtin_ieee_support_divide, &
    __builtin_ieee_support_flag, __builtin_ieee_support_halting, &
    __builtin_ieee_support_inf, __builtin_ieee_support_io, &
    __builtin_ieee_support_nan, __builtin_ieee_support_rounding, &
    __builtin_ieee_support_sqrt, &
    __builtin_ieee_support_standard, __builtin_ieee_support_subnormal, &
    __builtin_ieee_support_underflow_control

  type :: __force_derived_type_instantiations
    type(__builtin_c_ptr) :: c_ptr
    type(__builtin_c_funptr) :: c_funptr
    type(__builtin_event_type) :: event_type
    type(__builtin_lock_type) :: lock_type
    type(__builtin_team_type) :: team_type
  end type

  intrinsic :: __builtin_compiler_options, __builtin_compiler_version
  public :: __builtin_compiler_options, __builtin_compiler_version

  interface operator(==)
    module procedure __builtin_c_ptr_eq
  end interface
  public :: operator(==)

  interface operator(/=)
    module procedure __builtin_c_ptr_ne
  end interface
  public :: operator(/=)

  interface __builtin_c_associated
    module procedure c_associated_c_ptr
    module procedure c_associated_c_funptr
  end interface
  public :: __builtin_c_associated
!  private :: c_associated_c_ptr, c_associated_c_funptr

  type(__builtin_c_ptr), parameter, public :: __builtin_c_null_ptr = __builtin_c_ptr(0)
  type(__builtin_c_funptr), parameter, public :: &
    __builtin_c_null_funptr = __builtin_c_funptr(0)

  public :: __builtin_c_ptr_eq
  public :: __builtin_c_ptr_ne
  public :: __builtin_c_funloc

  contains

  elemental logical function __builtin_c_ptr_eq(x, y)
    type(__builtin_c_ptr), intent(in) :: x, y
    __builtin_c_ptr_eq = x%__address == y%__address
  end function

  elemental logical function __builtin_c_ptr_ne(x, y)
    type(__builtin_c_ptr), intent(in) :: x, y
    __builtin_c_ptr_ne = x%__address /= y%__address
  end function

  function __builtin_c_funloc(x)
    type(__builtin_c_funptr) :: __builtin_c_funloc
    external :: x
    __builtin_c_funloc = __builtin_c_funptr(loc(x))
  end function

  pure logical function c_associated_c_ptr(c_ptr_1, c_ptr_2)
    type(__builtin_c_ptr), intent(in) :: c_ptr_1
    type(__builtin_c_ptr), intent(in), optional :: c_ptr_2
    if (c_ptr_1%__address == __builtin_c_null_ptr%__address) then
      c_associated_c_ptr = .false.
    else if (present(c_ptr_2)) then
      c_associated_c_ptr = c_ptr_1%__address == c_ptr_2%__address
    else
      c_associated_c_ptr = .true.
    end if
  end function c_associated_c_ptr

  pure logical function c_associated_c_funptr(c_ptr_1, c_ptr_2)
    type(__builtin_c_funptr), intent(in) :: c_ptr_1
    type(__builtin_c_funptr), intent(in), optional :: c_ptr_2
    if (c_ptr_1%__address == __builtin_c_null_ptr%__address) then
      c_associated_c_funptr = .false.
    else if (present(c_ptr_2)) then
      c_associated_c_funptr = c_ptr_1%__address == c_ptr_2%__address
    else
      c_associated_c_funptr = .true.
    end if
  end function c_associated_c_funptr

end module
