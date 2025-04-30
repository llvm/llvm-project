!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This file is compiled with other test dllimport_test.f90.
! The dllimport_test.f90 relies on 'dllimport_test.f90_mod.f90' external module
! to test the presence of 'dllimport' storage class in LLVM IR
! for external module and opaque type.
!
! REQUIRES: system-windows
! RUN: true

module dllimport_module
  implicit none

  type t_type
    private
    integer :: a, b
  end type

  type(t_type), parameter :: array(2) = (/t_type(1, 1), t_type(1, 0)/)

  interface foobar
    module procedure test
  end interface

  contains
  subroutine test(a)
    type(t_type), dimension(:) :: a
    return
  end subroutine
end module
