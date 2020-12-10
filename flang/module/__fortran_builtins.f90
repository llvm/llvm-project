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

  use __Fortran_type_info, only: __builtin_c_ptr, __builtin_c_funptr
  integer, parameter, private :: int64 = selected_int_kind(18)

  intrinsic :: __builtin_c_f_pointer

  type :: __builtin_event_type
    integer(kind=int64) :: __count
  end type

  type :: __builtin_lock_type
    integer(kind=int64) :: __count
  end type

  type :: __builtin_team_type
    integer(kind=int64) :: __id
  end type
end module
