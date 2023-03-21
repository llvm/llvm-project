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

! fctid, fctidz, fctiw, fctiwz, fctudz, fctuwz
  abstract interface
    elemental real(4) function func_r4r4x(x)
      real(4), intent(in) :: x
    end function func_r4r4x
    elemental real(8) function func_r8r8x(x)
      real(8), intent(in) :: x
    end function func_r8r8x
  end interface

  procedure(func_r8r8x) :: __ppc_fctid
  interface fctid
    procedure :: __ppc_fctid
  end interface fctid
  public :: fctid

  procedure(func_r8r8x) :: __ppc_fctidz
  interface fctidz
    procedure :: __ppc_fctidz
  end interface fctidz
  public :: fctidz

  procedure(func_r8r8x) :: __ppc_fctiw
  interface fctiw
    procedure :: __ppc_fctiw
  end interface fctiw
  public :: fctiw

  procedure(func_r8r8x) :: __ppc_fctiwz
  interface fctiwz
    procedure :: __ppc_fctiwz
  end interface fctiwz
  public :: fctiwz

  procedure(func_r8r8x) :: __ppc_fctudz
  interface fctudz
    procedure :: __ppc_fctudz
  end interface fctudz
  public :: fctudz

  procedure(func_r8r8x) :: __ppc_fctuwz
  interface fctuwz
    procedure :: __ppc_fctuwz
  end interface fctuwz
  public :: fctuwz

! fcfi, fcfid, fcfud
  abstract interface
    elemental real(8) function func_r8r8i(i)
      real(8), intent(in) :: i
    end function func_r8r8i
  end interface

  procedure(func_r8r8i) :: __ppc_fcfi
  interface fcfi
    procedure :: __ppc_fcfi
  end interface fcfi
  public :: fcfi

  procedure(func_r8r8i) :: __ppc_fcfid
  interface fcfid
    procedure :: __ppc_fcfid
  end interface fcfid
  public :: fcfid

  procedure(func_r8r8i) :: __ppc_fcfud
  interface fcfud
    procedure :: __ppc_fcfud
  end interface fcfud
  public :: fcfud

! fnabs
  procedure(func_r4r4x) :: __ppc_fnabs_r4
  procedure(func_r8r8x) :: __ppc_fnabs_r8
  interface fnabs
    procedure :: __ppc_fnabs_r4
    procedure :: __ppc_fnabs_r8
  end interface fnabs
  public :: fnabs

! fre, fres
  procedure(func_r8r8x) :: __ppc_fre
  interface fre
    procedure :: __ppc_fre
  end interface fre
  public :: fre

  procedure(func_r4r4x) :: __ppc_fres
  interface fres
    procedure :: __ppc_fres
  end interface fres
  public :: fres

! frsqrte, frsqrtes
  procedure(func_r8r8x) :: __ppc_frsqrte
  interface frsqrte
    procedure :: __ppc_frsqrte
  end interface frsqrte
  public :: frsqrte

  procedure(func_r4r4x) :: __ppc_frsqrtes
  interface frsqrtes
    procedure :: __ppc_frsqrtes
  end interface frsqrtes
  public :: frsqrtes

! mtfsf, mtfsfi
  interface mtfsf
    subroutine __ppc_mtfsf(mask, r)
      integer(4), intent(in) :: mask
      real(8), intent(in) :: r
    end subroutine __ppc_mtfsf
  end interface mtfsf
  public :: mtfsf

  interface mtfsfi
    subroutine __ppc_mtfsfi(bf, i)
      integer(4), intent(in) :: bf
      integer(4), intent(in) :: i
    end subroutine __ppc_mtfsfi
  end interface mtfsfi
  public :: mtfsfi
end module __Fortran_PPC_intrinsics
