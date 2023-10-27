!===-- module/__fortran_builtins.f90 ---------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! These naming shenanigans prevent names from Fortran intrinsic modules
! from being usable on INTRINSIC statements, and force the program
! to USE the standard intrinsic modules in order to access the
! standard names of the procedures.
module __Fortran_builtins

  intrinsic :: __builtin_c_loc
  intrinsic :: __builtin_c_f_pointer
  intrinsic :: sizeof ! extension

  intrinsic :: selected_int_kind
  private :: selected_int_kind
  integer, parameter, private :: int64 = selected_int_kind(18)

  type, bind(c) :: __builtin_c_ptr
    integer(kind=int64), private :: __address
  end type

  type, bind(c) :: __builtin_c_funptr
    integer(kind=int64), private :: __address
  end type

  type :: __builtin_event_type
    integer(kind=int64), private :: __count
  end type

  type :: __builtin_lock_type
    integer(kind=int64), private :: __count
  end type

  type :: __builtin_team_type
    integer(kind=int64), private :: __id
  end type

  integer, parameter :: __builtin_atomic_int_kind = selected_int_kind(18)
  integer, parameter :: __builtin_atomic_logical_kind = __builtin_atomic_int_kind

  procedure(type(__builtin_c_ptr)) :: __builtin_c_loc

  type :: __builtin_dim3
    integer :: x=1, y=1, z=1
  end type
  type(__builtin_dim3) :: &
    __builtin_threadIdx, __builtin_blockDim, __builtin_blockIdx, __builtin_gridDim
  integer, parameter :: __builtin_warpsize = 32

  intrinsic :: __builtin_fma
  intrinsic :: __builtin_ieee_is_nan, __builtin_ieee_is_negative, &
    __builtin_ieee_is_normal
  intrinsic :: __builtin_ieee_next_after, __builtin_ieee_next_down, &
    __builtin_ieee_next_up
  intrinsic :: scale ! for ieee_scalb
  intrinsic :: __builtin_ieee_selected_real_kind
  intrinsic :: __builtin_ieee_support_datatype, &
    __builtin_ieee_support_denormal, __builtin_ieee_support_divide, &
    __builtin_ieee_support_inf, __builtin_ieee_support_io, &
    __builtin_ieee_support_nan, __builtin_ieee_support_sqrt, &
    __builtin_ieee_support_standard, __builtin_ieee_support_subnormal, &
    __builtin_ieee_support_underflow_control

  type, private :: __force_derived_type_instantiations
    type(__builtin_c_ptr) :: c_ptr
    type(__builtin_c_funptr) :: c_funptr
    type(__builtin_event_type) :: event_type
    type(__builtin_lock_type) :: lock_type
    type(__builtin_team_type) :: team_type
  end type

  intrinsic :: __builtin_compiler_options, __builtin_compiler_version

  interface operator(==)
    module procedure __builtin_c_ptr_eq
  end interface
  interface operator(/=)
    module procedure __builtin_c_ptr_eq
  end interface

  interface __builtin_c_associated
    module procedure c_associated_c_ptr
    module procedure c_associated_c_funptr
  end interface
  private :: c_associated_c_ptr, c_associated_c_funptr

  type(__builtin_c_ptr), parameter :: __builtin_c_null_ptr = __builtin_c_ptr(0)
  type(__builtin_c_funptr), parameter :: __builtin_c_null_funptr = __builtin_c_funptr(0)

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

  pure logical function c_associated_c_funptr(c_funptr_1, c_funptr_2)
    type(__builtin_c_funptr), intent(in) :: c_funptr_1
    type(__builtin_c_funptr), intent(in), optional :: c_funptr_2
    if (c_funptr_1%__address == __builtin_c_null_ptr%__address) then
      c_associated_c_funptr = .false.
    else if (present(c_funptr_2)) then
      c_associated_c_funptr = c_funptr_1%__address == c_funptr_2%__address
    else
      c_associated_c_funptr = .true.
    end if
  end function c_associated_c_funptr

end module
