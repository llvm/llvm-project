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
    integer(kind=int64) :: __address
  end type

  type, bind(c) :: __builtin_c_funptr
    integer(kind=int64) :: __address
  end type

  type :: __builtin_event_type
    integer(kind=int64) :: __count
  end type

  type :: __builtin_lock_type
    integer(kind=int64) :: __count
  end type

  type :: __builtin_team_type
    integer(kind=int64) :: __id
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

contains

  elemental logical function __builtin_c_ptr_eq(x, y)
    type(__builtin_c_ptr), intent(in) :: x, y
    __builtin_c_ptr_eq = x%__address == y%__address
  end function

  elemental logical function __builtin_c_ptr_ne(x, y)
    type(__builtin_c_ptr), intent(in) :: x, y
    __builtin_c_ptr_ne = x%__address /= y%__address
  end function

end module
