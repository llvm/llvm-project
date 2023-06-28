!===-- module/__ppc_intrinsics.f90 -----------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

module __ppc_intrinsics

  private

! fmadd, fmsub, fnmadd, fnmsub
  abstract interface
    elemental real(4) function func_r4r4r4r4(a, x, y)
      real(4), intent(in) :: a, x, y
    end function func_r4r4r4r4
    elemental real(8) function func_r8r8r8r8(a, x, y)
      real(8), intent(in) :: a, x, y
    end function func_r8r8r8r8

!--------------------
! Vector intrinsic
!--------------------
!! ================ 2 arguments function interface ================
! vector(i) function f(vector(i), vector(i))
#define ELEM_FUNC_VIVIVI(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND##vi##VKIND(arg1, arg2); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! vector(u) function f(vector(u), vector(u))
#define ELEM_FUNC_VUVUVU(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vu##VKIND##vu##VKIND(arg1, arg2); \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! vector(r) function f(vector(r), vector(r))
#define ELEM_FUNC_VRVRVR(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND(arg1, arg2); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

  ELEM_FUNC_VIVIVI(1) ELEM_FUNC_VIVIVI(2) ELEM_FUNC_VIVIVI(4) ELEM_FUNC_VIVIVI(8)
  ELEM_FUNC_VUVUVU(1) ELEM_FUNC_VUVUVU(2) ELEM_FUNC_VUVUVU(4) ELEM_FUNC_VUVUVU(8)
  ELEM_FUNC_VRVRVR(4) ELEM_FUNC_VRVRVR(8)

#undef ELEM_FUNC_VRVRVR
#undef ELEM_FUNC_VUVUVU
#undef ELEM_FUNC_VIVIVI

!! ================ 3 arguments function interface ================
! vector(r) function f(vector(r), vector(r), vector(r))
#define ELEM_FUNC_VRVRVRVR(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1, arg2, arg3; \
  end function ;

  ELEM_FUNC_VRVRVRVR(4) ELEM_FUNC_VRVRVRVR(8)

#undef ELEM_FUNC_VRVRVRVR

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

!---------------------------------
! vector function(vector, vector)
!---------------------------------
#define VI_VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND
#define VU_VU_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##vu##VKIND
#define VR_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND

#define VEC_VI_VI_VI(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##vi##VKIND) :: VI_VI_VI(NAME, VKIND);
#define VEC_VU_VU_VU(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND##vu##VKIND) :: VU_VU_VU(NAME, VKIND);
#define VEC_VR_VR_VR(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND) :: VR_VR_VR(NAME, VKIND);

! vec_add
  VEC_VI_VI_VI(vec_add,1) VEC_VI_VI_VI(vec_add,2) VEC_VI_VI_VI(vec_add,4) VEC_VI_VI_VI(vec_add,8)
  VEC_VU_VU_VU(vec_add,1) VEC_VU_VU_VU(vec_add,2) VEC_VU_VU_VU(vec_add,4) VEC_VU_VU_VU(vec_add,8)
  VEC_VR_VR_VR(vec_add,4) VEC_VR_VR_VR(vec_add,8)
  interface vec_add
    procedure :: VI_VI_VI(vec_add,1), VI_VI_VI(vec_add,2), VI_VI_VI(vec_add,4), VI_VI_VI(vec_add,8)
    procedure :: VU_VU_VU(vec_add,1), VU_VU_VU(vec_add,2), VU_VU_VU(vec_add,4), VU_VU_VU(vec_add,8)
    procedure :: VR_VR_VR(vec_add,4), VR_VR_VR(vec_add,8)
  end interface vec_add
  public :: vec_add

! vec_and
  VEC_VI_VI_VI(vec_and,1) VEC_VI_VI_VI(vec_and,2) VEC_VI_VI_VI(vec_and,4) VEC_VI_VI_VI(vec_and,8)
  VEC_VU_VU_VU(vec_and,1) VEC_VU_VU_VU(vec_and,2) VEC_VU_VU_VU(vec_and,4) VEC_VU_VU_VU(vec_and,8)
  VEC_VR_VR_VR(vec_and,4) VEC_VR_VR_VR(vec_and,8)
  interface vec_and
    procedure :: VI_VI_VI(vec_and,1), VI_VI_VI(vec_and,2), VI_VI_VI(vec_and,4), VI_VI_VI(vec_and,8)
    procedure :: VU_VU_VU(vec_and,1), VU_VU_VU(vec_and,2), VU_VU_VU(vec_and,4), VU_VU_VU(vec_and,8)
    procedure :: VR_VR_VR(vec_and,4), VR_VR_VR(vec_and,8)
  end interface vec_and
  public :: vec_and

! vec_max
  VEC_VI_VI_VI(vec_max,1) VEC_VI_VI_VI(vec_max,2) VEC_VI_VI_VI(vec_max,4) VEC_VI_VI_VI(vec_max,8)
  VEC_VU_VU_VU(vec_max,1) VEC_VU_VU_VU(vec_max,2) VEC_VU_VU_VU(vec_max,4) VEC_VU_VU_VU(vec_max,8)
  VEC_VR_VR_VR(vec_max,4) VEC_VR_VR_VR(vec_max,8)
  interface vec_max
    procedure :: VI_VI_VI(vec_max,1), VI_VI_VI(vec_max,2), VI_VI_VI(vec_max,4), VI_VI_VI(vec_max,8)
    procedure :: VU_VU_VU(vec_max,1), VU_VU_VU(vec_max,2), VU_VU_VU(vec_max,4), VU_VU_VU(vec_max,8)
    procedure :: VR_VR_VR(vec_max,4), VR_VR_VR(vec_max,8)
  end interface vec_max
  public :: vec_max

! vec_min
  VEC_VI_VI_VI(vec_min,1) VEC_VI_VI_VI(vec_min,2) VEC_VI_VI_VI(vec_min,4) VEC_VI_VI_VI(vec_min,8)
  VEC_VU_VU_VU(vec_min,1) VEC_VU_VU_VU(vec_min,2) VEC_VU_VU_VU(vec_min,4) VEC_VU_VU_VU(vec_min,8)
  VEC_VR_VR_VR(vec_min,4) VEC_VR_VR_VR(vec_min,8)
  interface vec_min
    procedure :: VI_VI_VI(vec_min,1), VI_VI_VI(vec_min,2), VI_VI_VI(vec_min,4), VI_VI_VI(vec_min,8)
    procedure :: VU_VU_VU(vec_min,1), VU_VU_VU(vec_min,2), VU_VU_VU(vec_min,4), VU_VU_VU(vec_min,8)
    procedure :: VR_VR_VR(vec_min,4), VR_VR_VR(vec_min,8)
  end interface vec_min
  public :: vec_min

! vec_mul
  VEC_VI_VI_VI(vec_mul,1) VEC_VI_VI_VI(vec_mul,2) VEC_VI_VI_VI(vec_mul,4) VEC_VI_VI_VI(vec_mul,8)
  VEC_VU_VU_VU(vec_mul,1) VEC_VU_VU_VU(vec_mul,2) VEC_VU_VU_VU(vec_mul,4) VEC_VU_VU_VU(vec_mul,8)
  VEC_VR_VR_VR(vec_mul,4) VEC_VR_VR_VR(vec_mul,8)
  interface vec_mul
    procedure :: VI_VI_VI(vec_mul,1), VI_VI_VI(vec_mul,2), VI_VI_VI(vec_mul,4), VI_VI_VI(vec_mul,8)
    procedure :: VU_VU_VU(vec_mul,1), VU_VU_VU(vec_mul,2), VU_VU_VU(vec_mul,4), VU_VU_VU(vec_mul,8)
    procedure :: VR_VR_VR(vec_mul,4), VR_VR_VR(vec_mul,8)
  end interface vec_mul
  public :: vec_mul

! vec_sub
  VEC_VI_VI_VI(vec_sub,1) VEC_VI_VI_VI(vec_sub,2) VEC_VI_VI_VI(vec_sub,4) VEC_VI_VI_VI(vec_sub,8)
  VEC_VU_VU_VU(vec_sub,1) VEC_VU_VU_VU(vec_sub,2) VEC_VU_VU_VU(vec_sub,4) VEC_VU_VU_VU(vec_sub,8)
  VEC_VR_VR_VR(vec_sub,4) VEC_VR_VR_VR(vec_sub,8)
  interface vec_sub
    procedure :: VI_VI_VI(vec_sub,1), VI_VI_VI(vec_sub,2), VI_VI_VI(vec_sub,4), VI_VI_VI(vec_sub,8)
    procedure :: VU_VU_VU(vec_sub,1), VU_VU_VU(vec_sub,2), VU_VU_VU(vec_sub,4), VU_VU_VU(vec_sub,8)
    procedure :: VR_VR_VR(vec_sub,4), VR_VR_VR(vec_sub,8)
  end interface vec_sub
  public :: vec_sub

! vec_xor
  VEC_VI_VI_VI(vec_xor,1) VEC_VI_VI_VI(vec_xor,2) VEC_VI_VI_VI(vec_xor,4) VEC_VI_VI_VI(vec_xor,8)
  VEC_VU_VU_VU(vec_xor,1) VEC_VU_VU_VU(vec_xor,2) VEC_VU_VU_VU(vec_xor,4) VEC_VU_VU_VU(vec_xor,8)
  VEC_VR_VR_VR(vec_xor,4) VEC_VR_VR_VR(vec_xor,8)
  interface vec_xor
    procedure :: VI_VI_VI(vec_xor,1), VI_VI_VI(vec_xor,2), VI_VI_VI(vec_xor,4), VI_VI_VI(vec_xor,8)
    procedure :: VU_VU_VU(vec_xor,1), VU_VU_VU(vec_xor,2), VU_VU_VU(vec_xor,4), VU_VU_VU(vec_xor,8)
    procedure :: VR_VR_VR(vec_xor,4), VR_VR_VR(vec_xor,8)
  end interface vec_xor
  public :: vec_xor

#undef VEC_VR_VR_VR
#undef VEC_VU_VU_VU
#undef VEC_VI_VI_VI
#undef VR_VR_VR
#undef VU_VU_VU
#undef VI_VI_VI

!-----------------------------------------
! vector function(vector, vector, vector)
!-----------------------------------------
#define VR_VR_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND

#define VEC_VR_VR_VR_VR(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND) :: VR_VR_VR_VR(NAME, VKIND);

! vec_madd
  VEC_VR_VR_VR_VR(vec_madd,4) VEC_VR_VR_VR_VR(vec_madd,8)
  interface vec_madd
    procedure :: VR_VR_VR_VR(vec_madd,4), VR_VR_VR_VR(vec_madd,8)
  end interface vec_madd
  public :: vec_madd

! vec_nmsub
  VEC_VR_VR_VR_VR(vec_nmsub,4) VEC_VR_VR_VR_VR(vec_nmsub,8)
  interface vec_nmsub
    procedure :: VR_VR_VR_VR(vec_nmsub,4), VR_VR_VR_VR(vec_nmsub,8)
  end interface vec_nmsub
  public :: vec_nmsub

#undef VEC_VR_VR_VR_VR
#undef VR_VR_VR_VR

end module __ppc_intrinsics
