!===-- module/__fortran_ppc_intrinsics.f90 ---------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

module __Fortran_PPC_intrinsics

  private

! fmadd, fmsub, fnmadd, fnmsub
  abstract interface
    elemental real(4) function func_r4r4r4r4(a, x, y)
      real(4), intent(in) :: a, x, y
    end function func_r4r4r4r4
    elemental real(8) function func_r8r8r8r8(a, x, y)
      real(8), intent(in) :: a, x, y
    end function func_r8r8r8r8
  end interface

  procedure(func_r4r4r4r4) :: __ppc_fmadd_r4
  procedure(func_r8r8r8r8) :: __ppc_fmadd_r8
  interface fmadd
    procedure :: __ppc_fmadd_r4
    procedure :: __ppc_fmadd_r8
  end interface fmadd
  public :: fmadd

  procedure(func_r4r4r4r4) :: __ppc_fmsub_r4
  procedure(func_r8r8r8r8) :: __ppc_fmsub_r8
  interface fmsub
    procedure :: __ppc_fmsub_r4
    procedure :: __ppc_fmsub_r8
  end interface fmsub
  public :: fmsub

  procedure(func_r4r4r4r4) :: __ppc_fnmadd_r4
  procedure(func_r8r8r8r8) :: __ppc_fnmadd_r8
  interface fnmadd
    procedure :: __ppc_fnmadd_r4
    procedure :: __ppc_fnmadd_r8
  end interface fnmadd
  public :: fnmadd

  procedure(func_r4r4r4r4) :: __ppc_fnmsub_r4
  procedure(func_r8r8r8r8) :: __ppc_fnmsub_r8
  interface fnmsub
    procedure :: __ppc_fnmsub_r4
    procedure :: __ppc_fnmsub_r8
  end interface fnmsub
  public :: fnmsub

end module __Fortran_PPC_intrinsics
