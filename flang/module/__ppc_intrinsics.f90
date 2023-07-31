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
!! ================ 1 argument function interface ================
! vector(i) function f(vector(i))
#define ELEM_FUNC_VIVI(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND(arg1); \
    vector(integer(VKIND)), intent(in) :: arg1; \
  end function ;

! vector(r) function f(vector(r))
#define ELEM_FUNC_VRVR_2(VKIND1, VKIND2) \
  elemental vector(real(VKIND1)) function elem_func_vr##VKIND1##vr##VKIND2(arg1); \
    vector(real(VKIND2)), intent(in) :: arg1; \
  end function ;
#define ELEM_FUNC_VRVR(VKIND) ELEM_FUNC_VRVR_2(VKIND, VKIND)

  ELEM_FUNC_VIVI(1) ELEM_FUNC_VIVI(2) ELEM_FUNC_VIVI(4) ELEM_FUNC_VIVI(8)
  ELEM_FUNC_VRVR_2(4,8) ELEM_FUNC_VRVR_2(8,4)
  ELEM_FUNC_VRVR(4) ELEM_FUNC_VRVR(8)

#undef ELEM_FUNC_VRVR
#undef ELEM_FUNC_VRVR_2
#undef ELEM_FUNC_VIVI

!! ================ 2 arguments function interface ================
! vector(i) function f(vector(i), vector(i))
#define ELEM_FUNC_VIVIVI(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND##vi##VKIND(arg1, arg2); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! vector(u) function f(vector(i), vector(i))
#define ELEM_FUNC_VUVIVI(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vi##VKIND##vi##VKIND(arg1, arg2); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! vector(i) function f(vector(i), vector(u))
#define ELEM_FUNC_VIVIVU_2(VKIND1, VKIND2) \
  elemental vector(integer(VKIND1)) function elem_func_vi##VKIND1##vi##VKIND1##vu##VKIND2(arg1, arg2); \
    vector(integer(VKIND1)), intent(in) :: arg1; \
    vector(unsigned(VKIND2)), intent(in) :: arg2; \
  end function ;
#define ELEM_FUNC_VIVIVU(VKIND) ELEM_FUNC_VIVIVU_2(VKIND, VKIND)

! vector(u) function f(vector(u), vector(u))
#define ELEM_FUNC_VUVUVU_2(VKIND1, VKIND2) \
  elemental vector(unsigned(VKIND1)) function elem_func_vu##VKIND1##vu##VKIND1##vu##VKIND2(arg1, arg2); \
    vector(unsigned(VKIND1)), intent(in) :: arg1; \
    vector(unsigned(VKIND2)), intent(in) :: arg2; \
  end function ;
#define ELEM_FUNC_VUVUVU(VKIND) ELEM_FUNC_VUVUVU_2(VKIND, VKIND)

! vector(r) function f(vector(r), vector(r))
#define ELEM_FUNC_VRVRVR(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND(arg1, arg2); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! vector(r) function f(vector(r), vector(u))
#define ELEM_FUNC_VRVRVU_2(VKIND1, VKIND2) \
  elemental vector(real(VKIND1)) function elem_func_vr##VKIND1##vr##VKIND1##vu##VKIND2(arg1, arg2); \
    vector(real(VKIND1)), intent(in) :: arg1; \
    vector(unsigned(VKIND2)), intent(in) :: arg2; \
  end function ;

! vector(u) function f(vector(r), vector(r))
#define ELEM_FUNC_VUVRVR(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vr##VKIND##vr##VKIND(arg1, arg2); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! integer function f(vector(i), vector(i))
#define ELEM_FUNC_IVIVI(RKIND, VKIND) \
  elemental integer(RKIND) function elem_func_i##RKIND##vi##VKIND##vi##VKIND(arg1, arg2); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! integer function f(vector(u), vector(u))
#define ELEM_FUNC_IVUVU(RKIND, VKIND) \
  elemental integer(RKIND) function elem_func_i##RKIND##vu##VKIND##vu##VKIND(arg1, arg2); \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! integer function f(vector(r), vector(r))
#define ELEM_FUNC_IVRVR(RKIND, VKIND) \
  elemental integer(RKIND) function elem_func_i##RKIND##vr##VKIND##vr##VKIND(arg1, arg2); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
  end function ;

! vector(r) function f(vector(i), i)
#define ELEM_FUNC_VRVII(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vi##VKIND##i(arg1, arg2); \
    vector(integer(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
  end function ;

! vector(r) function f(vector(u), i)
#define ELEM_FUNC_VRVUI(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vu##VKIND##i(arg1, arg2); \
    vector(unsigned(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
  end function ;

! The following macros are specific for the vec_convert(v, mold) intrinsics as
! the argument keywords are different from the other vector intrinsics.
!
! vector(i) function f(vector(i), vector(i))
#define FUNC_VEC_CONVERT_VIVIVI(VKIND) \
  pure vector(integer(VKIND)) function func_vec_convert_vi##VKIND##vi##vi##VKIND(v, mold); \
    vector(integer(8)), intent(in) :: v; \
    !dir$ ignore_tkr(tk) v; \
    vector(integer(VKIND)), intent(in) :: mold; \
    !dir$ ignore_tkr(r) mold; \
  end function ;

! vector(u) function f(vector(i), vector(u))
#define FUNC_VEC_CONVERT_VUVIVU(VKIND) \
  pure vector(unsigned(VKIND)) function func_vec_convert_vu##VKIND##vi##vu##VKIND(v, mold); \
    vector(integer(8)), intent(in) :: v; \
    !dir$ ignore_tkr(tk) v; \
    vector(unsigned(VKIND)), intent(in) :: mold; \
    !dir$ ignore_tkr(r) mold; \
  end function ;

! vector(r) function f(vector(i), vector(r))
#define FUNC_VEC_CONVERT_VRVIVR(VKIND) \
  pure vector(real(VKIND)) function func_vec_convert_vr##VKIND##vi##vr##VKIND(v, mold); \
    vector(integer(8)), intent(in) :: v; \
    !dir$ ignore_tkr(tk) v; \
    vector(real(VKIND)), intent(in) :: mold; \
    !dir$ ignore_tkr(r) mold; \
  end function ;

  FUNC_VEC_CONVERT_VIVIVI(1) FUNC_VEC_CONVERT_VIVIVI(2) FUNC_VEC_CONVERT_VIVIVI(4) FUNC_VEC_CONVERT_VIVIVI(8)
  FUNC_VEC_CONVERT_VUVIVU(1) FUNC_VEC_CONVERT_VUVIVU(2) FUNC_VEC_CONVERT_VUVIVU(4) FUNC_VEC_CONVERT_VUVIVU(8)
  FUNC_VEC_CONVERT_VRVIVR(4) FUNC_VEC_CONVERT_VRVIVR(8)

  ELEM_FUNC_VIVIVI(1) ELEM_FUNC_VIVIVI(2) ELEM_FUNC_VIVIVI(4) ELEM_FUNC_VIVIVI(8)
  ELEM_FUNC_VUVIVI(1) ELEM_FUNC_VUVIVI(2) ELEM_FUNC_VUVIVI(4) ELEM_FUNC_VUVIVI(8)
  ELEM_FUNC_VUVUVU(1) ELEM_FUNC_VUVUVU(2) ELEM_FUNC_VUVUVU(4) ELEM_FUNC_VUVUVU(8)
  ELEM_FUNC_VIVIVU(1) ELEM_FUNC_VIVIVU(2) ELEM_FUNC_VIVIVU(4) ELEM_FUNC_VIVIVU(8)
  ELEM_FUNC_VIVIVU_2(1,2) ELEM_FUNC_VIVIVU_2(1,4)
  ELEM_FUNC_VIVIVU_2(2,1) ELEM_FUNC_VIVIVU_2(2,4)
  ELEM_FUNC_VIVIVU_2(4,1) ELEM_FUNC_VIVIVU_2(4,2)
  ELEM_FUNC_VUVUVU_2(1,2) ELEM_FUNC_VUVUVU_2(1,4)
  ELEM_FUNC_VUVUVU_2(2,1) ELEM_FUNC_VUVUVU_2(2,4)
  ELEM_FUNC_VUVUVU_2(4,1) ELEM_FUNC_VUVUVU_2(4,2)
  ELEM_FUNC_VRVRVU_2(4,1) ELEM_FUNC_VRVRVU_2(4,2)
  ELEM_FUNC_VRVRVR(4) ELEM_FUNC_VRVRVR(8)
  ELEM_FUNC_VUVRVR(4) ELEM_FUNC_VUVRVR(8)
  ELEM_FUNC_IVIVI(4,1) ELEM_FUNC_IVIVI(4,2) ELEM_FUNC_IVIVI(4,4) ELEM_FUNC_IVIVI(4,8)
  ELEM_FUNC_IVUVU(4,1) ELEM_FUNC_IVUVU(4,2) ELEM_FUNC_IVUVU(4,4) ELEM_FUNC_IVUVU(4,8)
  ELEM_FUNC_IVRVR(4,4) ELEM_FUNC_IVRVR(4,8)
  ELEM_FUNC_VRVII(4) ELEM_FUNC_VRVII(8)
  ELEM_FUNC_VRVUI(4) ELEM_FUNC_VRVUI(8)

#undef FUNC_VEC_CONVERT_VRVIVR
#undef FUNC_VEC_CONVERT_VUVIVU
#undef FUNC_VEC_CONVERT_VIVIVI
#undef ELEM_FUNC_VRVUI
#undef ELEM_FUNC_VRVII
#undef ELEM_FUNC_IVIVI
#undef ELEM_FUNC_IVUVU
#undef ELEM_FUNC_VIVIVU_2
#undef ELEM_FUNC_VUVUVU_2
#undef ELEM_FUNC_VRVRVU_2
#undef ELEM_FUNC_IVRVR
#undef ELEM_FUNC_VUVRVR
#undef ELEM_FUNC_VRVRVR
#undef ELEM_FUNC_VIVIVU
#undef ELEM_FUNC_VUVUVU
#undef ELEM_FUNC_VUVIVI
#undef ELEM_FUNC_VIVIVI

!! ================ 3 arguments function interface ================
! vector(r) function f(vector(r), vector(r), vector(r))
#define ELEM_FUNC_VRVRVRVR(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1, arg2, arg3; \
  end function ;

! vector(i) function f(vector(i), vector(i), vector(u))
#define ELEM_FUNC_VIVIVIVU(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND##vi##VKIND##vu##VKIND(arg1, arg2, arg3); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
    vector(unsigned(VKIND)), intent(in) :: arg3; \
  end function ;

! vector(u) function f(vector(u), vector(u), vector(u))
#define ELEM_FUNC_VUVUVUVU(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vu##VKIND##vu##VKIND##vu##VKIND(arg1, arg2, arg3); \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2, arg3; \
  end function ;

! vector(r) function f(vector(r), vector(r), vector(u))
#define ELEM_FUNC_VRVRVRVU(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vu##VKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
    vector(unsigned(VKIND)), intent(in) :: arg3; \
  end function ;


! vector(i) function f(vector(i), vector(i), i)
#define ELEM_FUNC_VIVIVII(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND##vi##VKIND##i(arg1, arg2, arg3); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
    integer(8), intent(in) :: arg3; \
    !dir$ ignore_tkr(k) arg3; \
  end function ;

! vector(u) function f(vector(u), vector(u), i)
#define ELEM_FUNC_VUVUVUI(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vu##VKIND##vu##VKIND##i(arg1, arg2, arg3); \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2; \
    integer(8), intent(in) :: arg3; \
    !dir$ ignore_tkr(k) arg3; \
  end function ;

! vector(r) function f(vector(r), vector(r), i)
#define ELEM_FUNC_VRVRVRI(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND##i(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
    integer(8), intent(in) :: arg3; \
    !dir$ ignore_tkr(k) arg3; \
  end function ;

  ELEM_FUNC_VIVIVIVU(1) ELEM_FUNC_VIVIVIVU(2) ELEM_FUNC_VIVIVIVU(4) ELEM_FUNC_VIVIVIVU(8)
  ELEM_FUNC_VUVUVUVU(1) ELEM_FUNC_VUVUVUVU(2) ELEM_FUNC_VUVUVUVU(4) ELEM_FUNC_VUVUVUVU(8)
  ELEM_FUNC_VRVRVRVU(4) ELEM_FUNC_VRVRVRVU(8)
  ELEM_FUNC_VRVRVRVR(4) ELEM_FUNC_VRVRVRVR(8)
  ELEM_FUNC_VIVIVII(1) ELEM_FUNC_VIVIVII(2) ELEM_FUNC_VIVIVII(4) ELEM_FUNC_VIVIVII(8)
  ELEM_FUNC_VUVUVUI(1) ELEM_FUNC_VUVUVUI(2) ELEM_FUNC_VUVUVUI(4) ELEM_FUNC_VUVUVUI(8)
  ELEM_FUNC_VRVRVRI(4) ELEM_FUNC_VRVRVRI(8)

#undef ELEM_FUNC_VIVIVII
#undef ELEM_FUNC_VUVUVUI
#undef ELEM_FUNC_VRVRVRI
#undef ELEM_FUNC_VRVRVRVR
#undef ELEM_FUNC_VRVRVRVU
#undef ELEM_FUNC_VRVRVRVR
#undef ELEM_FUNC_VUVUVUVU
#undef ELEM_FUNC_VIVIVIVU

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

!-------------------------
! vector function(vector)
!-------------------------
#define VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND
#define VR_VR_2(NAME, VKIND1, VKIND2) __ppc_##NAME##_vr##VKIND1##vr##VKIND2
#define VR_VR(NAME, VKIND) VR_VR_2(NAME, VKIND, VKIND)

#define VEC_VI_VI(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND) :: VI_VI(NAME, VKIND);
#define VEC_VR_VR_2(NAME, VKIND1, VKIND2) \
  procedure(elem_func_vr##VKIND1##vr##VKIND2) :: VR_VR_2(NAME, VKIND1, VKIND2);
#define VEC_VR_VR(NAME, VKIND) VEC_VR_VR_2(NAME, VKIND, VKIND)

! vec_abs
  VEC_VI_VI(vec_abs,1) VEC_VI_VI(vec_abs,2) VEC_VI_VI(vec_abs,4) VEC_VI_VI(vec_abs,8)
  VEC_VR_VR(vec_abs,4) VEC_VR_VR(vec_abs,8)
  interface vec_abs
    procedure :: VI_VI(vec_abs,1), VI_VI(vec_abs,2), VI_VI(vec_abs,4), VI_VI(vec_abs,8)
    procedure :: VR_VR(vec_abs,4), VR_VR(vec_abs,8)
  end interface vec_abs
  public :: vec_abs

! vec_cvf
  VEC_VR_VR_2(vec_cvf,4,8) VEC_VR_VR_2(vec_cvf,8,4)
  interface vec_cvf
    procedure :: VR_VR_2(vec_cvf,4,8), VR_VR_2(vec_cvf,8,4)
  end interface vec_cvf
  public :: vec_cvf

#undef VEC_VR_VR
#undef VEC_VR_VR_2
#undef VEC_VI_VI
#undef VR_VR
#undef VR_VR_2
#undef VI_VI
  
!---------------------------------
! vector function(vector, vector)
!---------------------------------
#define VI_VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND
#define VU_VI_VI(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vi##VKIND##vi##VKIND
#define VU_VU_VU_2(NAME, VKIND1, VKIND2) __ppc_##NAME##_vu##VKIND1##vu##VKIND1##vu##VKIND2
#define VU_VU_VU(NAME, VKIND) VU_VU_VU_2(NAME, VKIND, VKIND)
#define VI_VI_VU_2(NAME, VKIND1, VKIND2) __ppc_##NAME##_vi##VKIND1##vi##VKIND1##vu##VKIND2
#define VI_VI_VU(NAME, VKIND) VI_VI_VU_2(NAME, VKIND, VKIND)
#define VR_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND
#define VR_VR_VU_2(NAME, VKIND1, VKIND2) __ppc_##NAME##_vr##VKIND1##vr##VKIND1##vu##VKIND2
#define VU_VR_VR(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vr##VKIND##vr##VKIND

#define VEC_VI_VI_VI(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##vi##VKIND) :: VI_VI_VI(NAME, VKIND);
#define VEC_VU_VI_VI(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vi##VKIND##vi##VKIND) :: VU_VI_VI(NAME, VKIND);
#define VEC_VU_VU_VU_2(NAME, VKIND1, VKIND2) \
  procedure(elem_func_vu##VKIND1##vu##VKIND1##vu##VKIND2) :: VU_VU_VU_2(NAME, VKIND1, VKIND2);
#define VEC_VU_VU_VU(NAME, VKIND) VEC_VU_VU_VU_2(NAME, VKIND, VKIND)
#define VEC_VI_VI_VU_2(NAME, VKIND1, VKIND2) \
  procedure(elem_func_vi##VKIND1##vi##VKIND1##vu##VKIND2) :: VI_VI_VU_2(NAME, VKIND1, VKIND2);
#define VEC_VI_VI_VU(NAME, VKIND) VEC_VI_VI_VU_2(NAME, VKIND, VKIND)
#define VEC_VR_VR_VR(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND) :: VR_VR_VR(NAME, VKIND);
#define VEC_VU_VR_VR(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vr##VKIND##vr##VKIND) :: VU_VR_VR(NAME, VKIND);
#define VEC_VR_VR_VU(NAME, VKIND1, VKIND2) \
  procedure(elem_func_vr##VKIND1##vr##VKIND1##vu##VKIND2) :: VR_VR_VU_2(NAME, VKIND1, VKIND2);

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

! vec_cmpge
  VEC_VU_VI_VI(vec_cmpge,1) VEC_VU_VI_VI(vec_cmpge,2) VEC_VU_VI_VI(vec_cmpge,4) VEC_VU_VI_VI(vec_cmpge,8)
  VEC_VU_VU_VU(vec_cmpge,1) VEC_VU_VU_VU(vec_cmpge,2) VEC_VU_VU_VU(vec_cmpge,4) VEC_VU_VU_VU(vec_cmpge,8)
  VEC_VU_VR_VR(vec_cmpge,4) VEC_VU_VR_VR(vec_cmpge,8)
  interface vec_cmpge
    procedure :: VU_VI_VI(vec_cmpge,1), VU_VI_VI(vec_cmpge,2), VU_VI_VI(vec_cmpge,4), VU_VI_VI(vec_cmpge,8)
    procedure :: VU_VU_VU(vec_cmpge,1), VU_VU_VU(vec_cmpge,2), VU_VU_VU(vec_cmpge,4), VU_VU_VU(vec_cmpge,8)
    procedure :: VU_VR_VR(vec_cmpge,4), VU_VR_VR(vec_cmpge,8)
  end interface vec_cmpge
  public :: vec_cmpge

! vec_cmpgt
  VEC_VU_VI_VI(vec_cmpgt,1) VEC_VU_VI_VI(vec_cmpgt,2) VEC_VU_VI_VI(vec_cmpgt,4) VEC_VU_VI_VI(vec_cmpgt,8)
  VEC_VU_VU_VU(vec_cmpgt,1) VEC_VU_VU_VU(vec_cmpgt,2) VEC_VU_VU_VU(vec_cmpgt,4) VEC_VU_VU_VU(vec_cmpgt,8)
  VEC_VU_VR_VR(vec_cmpgt,4) VEC_VU_VR_VR(vec_cmpgt,8)
  interface vec_cmpgt
    procedure :: VU_VI_VI(vec_cmpgt,1), VU_VI_VI(vec_cmpgt,2), VU_VI_VI(vec_cmpgt,4), VU_VI_VI(vec_cmpgt,8)
    procedure :: VU_VU_VU(vec_cmpgt,1), VU_VU_VU(vec_cmpgt,2), VU_VU_VU(vec_cmpgt,4), VU_VU_VU(vec_cmpgt,8)
    procedure :: VU_VR_VR(vec_cmpgt,4), VU_VR_VR(vec_cmpgt,8)
  end interface vec_cmpgt
  public :: vec_cmpgt

! vec_cmple
  VEC_VU_VI_VI(vec_cmple,1) VEC_VU_VI_VI(vec_cmple,2) VEC_VU_VI_VI(vec_cmple,4) VEC_VU_VI_VI(vec_cmple,8)
  VEC_VU_VU_VU(vec_cmple,1) VEC_VU_VU_VU(vec_cmple,2) VEC_VU_VU_VU(vec_cmple,4) VEC_VU_VU_VU(vec_cmple,8)
  VEC_VU_VR_VR(vec_cmple,4) VEC_VU_VR_VR(vec_cmple,8)
  interface vec_cmple
    procedure :: VU_VI_VI(vec_cmple,1), VU_VI_VI(vec_cmple,2), VU_VI_VI(vec_cmple,4), VU_VI_VI(vec_cmple,8)
    procedure :: VU_VU_VU(vec_cmple,1), VU_VU_VU(vec_cmple,2), VU_VU_VU(vec_cmple,4), VU_VU_VU(vec_cmple,8)
    procedure :: VU_VR_VR(vec_cmple,4), VU_VR_VR(vec_cmple,8)
  end interface vec_cmple
  public :: vec_cmple

! vec_cmplt
  VEC_VU_VI_VI(vec_cmplt,1) VEC_VU_VI_VI(vec_cmplt,2) VEC_VU_VI_VI(vec_cmplt,4) VEC_VU_VI_VI(vec_cmplt,8)
  VEC_VU_VU_VU(vec_cmplt,1) VEC_VU_VU_VU(vec_cmplt,2) VEC_VU_VU_VU(vec_cmplt,4) VEC_VU_VU_VU(vec_cmplt,8)
  VEC_VU_VR_VR(vec_cmplt,4) VEC_VU_VR_VR(vec_cmplt,8)
  interface vec_cmplt
    procedure :: VU_VI_VI(vec_cmplt,1), VU_VI_VI(vec_cmplt,2), VU_VI_VI(vec_cmplt,4), VU_VI_VI(vec_cmplt,8)
    procedure :: VU_VU_VU(vec_cmplt,1), VU_VU_VU(vec_cmplt,2), VU_VU_VU(vec_cmplt,4), VU_VU_VU(vec_cmplt,8)
    procedure :: VU_VR_VR(vec_cmplt,4), VU_VR_VR(vec_cmplt,8)
  end interface vec_cmplt
  public :: vec_cmplt

! vec_convert
! Argument 'v' has the `ignore_tkr` directive
#define CONVERT_VI_VI_VI(VKIND) __ppc_vec_convert_vi##VKIND##vi##vi##VKIND
#define CONVERT_VU_VI_VU(VKIND) __ppc_vec_convert_vu##VKIND##vi##vu##VKIND
#define CONVERT_VR_VI_VR(VKIND) __ppc_vec_convert_vr##VKIND##vi##vr##VKIND

#define VEC_CONVERT_VI_VI_VI(VKIND) \
  procedure(func_vec_convert_vi##VKIND##vi##vi##VKIND) :: CONVERT_VI_VI_VI(VKIND);
#define VEC_CONVERT_VU_VI_VU(VKIND) \
  procedure(func_vec_convert_vu##VKIND##vi##vu##VKIND) :: CONVERT_VU_VI_VU(VKIND);
#define VEC_CONVERT_VR_VI_VR(VKIND) \
  procedure(func_vec_convert_vr##VKIND##vi##vr##VKIND) :: CONVERT_VR_VI_VR(VKIND);

  VEC_CONVERT_VI_VI_VI(1) VEC_CONVERT_VI_VI_VI(2) VEC_CONVERT_VI_VI_VI(4) VEC_CONVERT_VI_VI_VI(8)
  VEC_CONVERT_VU_VI_VU(1) VEC_CONVERT_VU_VI_VU(2) VEC_CONVERT_VU_VI_VU(4) VEC_CONVERT_VU_VI_VU(8)
  VEC_CONVERT_VR_VI_VR(4) VEC_CONVERT_VR_VI_VR(8)
  interface vec_convert
    procedure :: CONVERT_VI_VI_VI(1), CONVERT_VI_VI_VI(2), CONVERT_VI_VI_VI(4), CONVERT_VI_VI_VI(8)
    procedure :: CONVERT_VU_VI_VU(1), CONVERT_VU_VI_VU(2), CONVERT_VU_VI_VU(4), CONVERT_VU_VI_VU(8)
    procedure :: CONVERT_VR_VI_VR(4), CONVERT_VR_VI_VR(8)
  end interface vec_convert
  public :: vec_convert

#undef VEC_CONVERT_VR_VI_VR
#undef VEC_CONVERT_VU_VI_VU
#undef VEC_CONVERT_VI_VI_VI
#undef CONVERT_VR_VI_VR
#undef CONVERT_VU_VI_VU
#undef CONVERT_VI_VI_VI

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

! vec_sl
  VEC_VI_VI_VU(vec_sl,1) VEC_VI_VI_VU(vec_sl,2) VEC_VI_VI_VU(vec_sl,4) VEC_VI_VI_VU(vec_sl,8)
  VEC_VU_VU_VU(vec_sl,1) VEC_VU_VU_VU(vec_sl,2) VEC_VU_VU_VU(vec_sl,4) VEC_VU_VU_VU(vec_sl,8)
  interface vec_sl
    procedure :: VI_VI_VU(vec_sl,1), VI_VI_VU(vec_sl,2), VI_VI_VU(vec_sl,4), VI_VI_VU(vec_sl,8)
    procedure :: VU_VU_VU(vec_sl,1), VU_VU_VU(vec_sl,2), VU_VU_VU(vec_sl,4), VU_VU_VU(vec_sl,8)
  end interface vec_sl
  public :: vec_sl

! vec_sll
  VEC_VI_VI_VU_2(vec_sll,1,1) VEC_VI_VI_VU_2(vec_sll,2,1) VEC_VI_VI_VU_2(vec_sll,4,1)
  VEC_VI_VI_VU_2(vec_sll,1,2) VEC_VI_VI_VU_2(vec_sll,2,2) VEC_VI_VI_VU_2(vec_sll,4,2)
  VEC_VI_VI_VU_2(vec_sll,1,4) VEC_VI_VI_VU_2(vec_sll,2,4) VEC_VI_VI_VU_2(vec_sll,4,4)
  VEC_VU_VU_VU_2(vec_sll,1,1) VEC_VU_VU_VU_2(vec_sll,2,1) VEC_VU_VU_VU_2(vec_sll,4,1)
  VEC_VU_VU_VU_2(vec_sll,1,2) VEC_VU_VU_VU_2(vec_sll,2,2) VEC_VU_VU_VU_2(vec_sll,4,2)
  VEC_VU_VU_VU_2(vec_sll,1,4) VEC_VU_VU_VU_2(vec_sll,2,4) VEC_VU_VU_VU_2(vec_sll,4,4)
  interface vec_sll
    procedure :: VI_VI_VU_2(vec_sll,1,1), VI_VI_VU_2(vec_sll,2,1), VI_VI_VU_2(vec_sll,4,1)
    procedure :: VI_VI_VU_2(vec_sll,1,2), VI_VI_VU_2(vec_sll,2,2), VI_VI_VU_2(vec_sll,4,2)
    procedure :: VI_VI_VU_2(vec_sll,1,4), VI_VI_VU_2(vec_sll,2,4), VI_VI_VU_2(vec_sll,4,4)
    procedure :: VU_VU_VU_2(vec_sll,1,1), VU_VU_VU_2(vec_sll,2,1), VU_VU_VU_2(vec_sll,4,1)
    procedure :: VU_VU_VU_2(vec_sll,1,2), VU_VU_VU_2(vec_sll,2,2), VU_VU_VU_2(vec_sll,4,2)
    procedure :: VU_VU_VU_2(vec_sll,1,4), VU_VU_VU_2(vec_sll,2,4), VU_VU_VU_2(vec_sll,4,4)
  end interface vec_sll
  public :: vec_sll

! vec_slo
  VEC_VI_VI_VU_2(vec_slo,1,1) VEC_VI_VI_VU_2(vec_slo,2,1) VEC_VI_VI_VU_2(vec_slo,4,1)
  VEC_VI_VI_VU_2(vec_slo,1,2) VEC_VI_VI_VU_2(vec_slo,2,2) VEC_VI_VI_VU_2(vec_slo,4,2)
  VEC_VU_VU_VU_2(vec_slo,1,1) VEC_VU_VU_VU_2(vec_slo,2,1) VEC_VU_VU_VU_2(vec_slo,4,1)
  VEC_VU_VU_VU_2(vec_slo,1,2) VEC_VU_VU_VU_2(vec_slo,2,2) VEC_VU_VU_VU_2(vec_slo,4,2)
  VEC_VR_VR_VU(vec_slo,4,1) VEC_VR_VR_VU(vec_slo,4,2)
  interface vec_slo
    procedure :: VI_VI_VU_2(vec_slo,1,1), VI_VI_VU_2(vec_slo,2,1), VI_VI_VU_2(vec_slo,4,1)
    procedure :: VI_VI_VU_2(vec_slo,1,2), VI_VI_VU_2(vec_slo,2,2), VI_VI_VU_2(vec_slo,4,2)
    procedure :: VU_VU_VU_2(vec_slo,1,1), VU_VU_VU_2(vec_slo,2,1), VU_VU_VU_2(vec_slo,4,1)
    procedure :: VU_VU_VU_2(vec_slo,1,2), VU_VU_VU_2(vec_slo,2,2), VU_VU_VU_2(vec_slo,4,2)
    procedure :: VR_VR_VU_2(vec_slo,4,1), VR_VR_VU_2(vec_slo,4,2)
  end interface vec_slo
  public :: vec_slo

! vec_sr
  VEC_VI_VI_VU(vec_sr,1) VEC_VI_VI_VU(vec_sr,2) VEC_VI_VI_VU(vec_sr,4) VEC_VI_VI_VU(vec_sr,8)
  VEC_VU_VU_VU(vec_sr,1) VEC_VU_VU_VU(vec_sr,2) VEC_VU_VU_VU(vec_sr,4) VEC_VU_VU_VU(vec_sr,8)
  interface vec_sr
    procedure :: VI_VI_VU(vec_sr,1), VI_VI_VU(vec_sr,2), VI_VI_VU(vec_sr,4), VI_VI_VU(vec_sr,8)
    procedure :: VU_VU_VU(vec_sr,1), VU_VU_VU(vec_sr,2), VU_VU_VU(vec_sr,4), VU_VU_VU(vec_sr,8)
  end interface vec_sr
  public :: vec_sr

! vec_srl
  VEC_VI_VI_VU_2(vec_srl,1,1) VEC_VI_VI_VU_2(vec_srl,2,1) VEC_VI_VI_VU_2(vec_srl,4,1)
  VEC_VI_VI_VU_2(vec_srl,1,2) VEC_VI_VI_VU_2(vec_srl,2,2) VEC_VI_VI_VU_2(vec_srl,4,2)
  VEC_VI_VI_VU_2(vec_srl,1,4) VEC_VI_VI_VU_2(vec_srl,2,4) VEC_VI_VI_VU_2(vec_srl,4,4)
  VEC_VU_VU_VU_2(vec_srl,1,1) VEC_VU_VU_VU_2(vec_srl,2,1) VEC_VU_VU_VU_2(vec_srl,4,1)
  VEC_VU_VU_VU_2(vec_srl,1,2) VEC_VU_VU_VU_2(vec_srl,2,2) VEC_VU_VU_VU_2(vec_srl,4,2)
  VEC_VU_VU_VU_2(vec_srl,1,4) VEC_VU_VU_VU_2(vec_srl,2,4) VEC_VU_VU_VU_2(vec_srl,4,4)
  interface vec_srl
    procedure :: VI_VI_VU_2(vec_srl,1,1), VI_VI_VU_2(vec_srl,2,1), VI_VI_VU_2(vec_srl,4,1)
    procedure :: VI_VI_VU_2(vec_srl,1,2), VI_VI_VU_2(vec_srl,2,2), VI_VI_VU_2(vec_srl,4,2)
    procedure :: VI_VI_VU_2(vec_srl,1,4), VI_VI_VU_2(vec_srl,2,4), VI_VI_VU_2(vec_srl,4,4)
    procedure :: VU_VU_VU_2(vec_srl,1,1), VU_VU_VU_2(vec_srl,2,1), VU_VU_VU_2(vec_srl,4,1)
    procedure :: VU_VU_VU_2(vec_srl,1,2), VU_VU_VU_2(vec_srl,2,2), VU_VU_VU_2(vec_srl,4,2)
    procedure :: VU_VU_VU_2(vec_srl,1,4), VU_VU_VU_2(vec_srl,2,4), VU_VU_VU_2(vec_srl,4,4)
  end interface vec_srl
  public :: vec_srl

! vec_sro
  VEC_VI_VI_VU_2(vec_sro,1,1) VEC_VI_VI_VU_2(vec_sro,2,1) VEC_VI_VI_VU_2(vec_sro,4,1)
  VEC_VI_VI_VU_2(vec_sro,1,2) VEC_VI_VI_VU_2(vec_sro,2,2) VEC_VI_VI_VU_2(vec_sro,4,2)
  VEC_VU_VU_VU_2(vec_sro,1,1) VEC_VU_VU_VU_2(vec_sro,2,1) VEC_VU_VU_VU_2(vec_sro,4,1)
  VEC_VU_VU_VU_2(vec_sro,1,2) VEC_VU_VU_VU_2(vec_sro,2,2) VEC_VU_VU_VU_2(vec_sro,4,2)
  VEC_VR_VR_VU(vec_sro,4,1) VEC_VR_VR_VU(vec_sro,4,2)
  interface vec_sro
    procedure :: VI_VI_VU_2(vec_sro,1,1), VI_VI_VU_2(vec_sro,2,1), VI_VI_VU_2(vec_sro,4,1)
    procedure :: VI_VI_VU_2(vec_sro,1,2), VI_VI_VU_2(vec_sro,2,2), VI_VI_VU_2(vec_sro,4,2)
    procedure :: VU_VU_VU_2(vec_sro,1,1), VU_VU_VU_2(vec_sro,2,1), VU_VU_VU_2(vec_sro,4,1)
    procedure :: VU_VU_VU_2(vec_sro,1,2), VU_VU_VU_2(vec_sro,2,2), VU_VU_VU_2(vec_sro,4,2)
    procedure :: VR_VR_VU_2(vec_sro,4,1), VR_VR_VU_2(vec_sro,4,2)
  end interface vec_sro
  public :: vec_sro

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

#undef VEC_VU_VR_VR
#undef VEC_VR_VR_VR
#undef VEC_VU_VU_VU
#undef VEC_VU_VU_VU_2
#undef VEC_VI_VI_VI
#undef VEC_VU_VI_VI
#undef VEC_VI_VI_VU
#undef VEC_VI_VI_VU_2
#undef VU_VR_VR
#undef VR_VR_VU_2
#undef VR_VR_VR
#undef VU_VU_VU
#undef VU_VU_VU_2
#undef VI_VI_VU
#undef VI_VI_VU_2
#undef VU_VI_VI
#undef VI_VI_VI

!-----------------------------------------
! vector function(vector, vector, vector)
!-----------------------------------------
#define VR_VR_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND
#define VI_VI_VI_VU(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND##vu##VKIND
#define VU_VU_VU_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##vu##VKIND##vu##VKIND
#define VR_VR_VR_VU(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##vu##VKIND

#define VEC_VR_VR_VR_VR(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND) :: VR_VR_VR_VR(NAME, VKIND);
#define VEC_VI_VI_VI_VU(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##vi##VKIND##vu##VKIND) :: VI_VI_VI_VU(NAME, VKIND);
#define VEC_VU_VU_VU_VU(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND##vu##VKIND##vu##VKIND) :: VU_VU_VU_VU(NAME, VKIND);
#define VEC_VR_VR_VR_VU(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vu##VKIND) :: VR_VR_VR_VU(NAME, VKIND);

! vec_madd
  VEC_VR_VR_VR_VR(vec_madd,4) VEC_VR_VR_VR_VR(vec_madd,8)
  interface vec_madd
    procedure :: VR_VR_VR_VR(vec_madd,4), VR_VR_VR_VR(vec_madd,8)
  end interface vec_madd
  public :: vec_madd

! vec_msub
  VEC_VR_VR_VR_VR(vec_msub,4) VEC_VR_VR_VR_VR(vec_msub,8)
  interface vec_msub
    procedure :: VR_VR_VR_VR(vec_msub,4), VR_VR_VR_VR(vec_msub,8)
  end interface vec_msub
  public :: vec_msub

! vec_nmadd
  VEC_VR_VR_VR_VR(vec_nmadd,4) VEC_VR_VR_VR_VR(vec_nmadd,8)
  interface vec_nmadd
    procedure :: VR_VR_VR_VR(vec_nmadd,4), VR_VR_VR_VR(vec_nmadd,8)
  end interface vec_nmadd
  public :: vec_nmadd

! vec_nmsub
  VEC_VR_VR_VR_VR(vec_nmsub,4) VEC_VR_VR_VR_VR(vec_nmsub,8)
  interface vec_nmsub
    procedure :: VR_VR_VR_VR(vec_nmsub,4), VR_VR_VR_VR(vec_nmsub,8)
  end interface vec_nmsub
  public :: vec_nmsub

! vec_sel
  VEC_VI_VI_VI_VU(vec_sel,1) VEC_VI_VI_VI_VU(vec_sel,2) VEC_VI_VI_VI_VU(vec_sel,4) VEC_VI_VI_VI_VU(vec_sel,8)
  VEC_VU_VU_VU_VU(vec_sel,1) VEC_VU_VU_VU_VU(vec_sel,2) VEC_VU_VU_VU_VU(vec_sel,4) VEC_VU_VU_VU_VU(vec_sel,8)
  VEC_VR_VR_VR_VU(vec_sel,4) VEC_VR_VR_VR_VU(vec_sel,8)
  interface vec_sel
    procedure :: VI_VI_VI_VU(vec_sel,1), VI_VI_VI_VU(vec_sel,2), VI_VI_VI_VU(vec_sel,4), VI_VI_VI_VU(vec_sel,8)
    procedure :: VU_VU_VU_VU(vec_sel,1), VU_VU_VU_VU(vec_sel,2), VU_VU_VU_VU(vec_sel,4), VU_VU_VU_VU(vec_sel,8)
    procedure :: VR_VR_VR_VU(vec_sel,4), VR_VR_VR_VU(vec_sel,8)
  end interface vec_sel
  public :: vec_sel

#undef VEC_VI_VI_VI_VU
#undef VEC_VU_VU_VU_VU
#undef VEC_VR_VR_VR_VU
#undef VEC_VR_VR_VR_VR
#undef VI_VI_VI_VU
#undef VU_VU_VU_VU
#undef VR_VR_VR_VU
#undef VR_VR_VR_VR

!----------------------------------
! integer function(vector, vector)
!----------------------------------
#define I_VI_VI(NAME, RKIND, VKIND) __ppc_##NAME##_i##RKIND##vi##VKIND##vi##VKIND
#define I_VU_VU(NAME, RKIND, VKIND) __ppc_##NAME##_i##RKIND##vu##VKIND##vu##VKIND
#define I_VR_VR(NAME, RKIND, VKIND) __ppc_##NAME##_i##RKIND##vr##VKIND##vr##VKIND

#define VEC_I_VI_VI(NAME, RKIND, VKIND) \
  procedure(elem_func_i##RKIND##vi##VKIND##vi##VKIND) :: I_VI_VI(NAME, RKIND, VKIND);
#define VEC_I_VU_VU(NAME, RKIND, VKIND) \
  procedure(elem_func_i##RKIND##vu##VKIND##vu##VKIND) :: I_VU_VU(NAME, RKIND, VKIND);
#define VEC_I_VR_VR(NAME, RKIND, VKIND) \
  procedure(elem_func_i##RKIND##vr##VKIND##vr##VKIND) :: I_VR_VR(NAME, RKIND, VKIND);

! vec_any_ge
  VEC_I_VI_VI(vec_any_ge,4,1) VEC_I_VI_VI(vec_any_ge,4,2) VEC_I_VI_VI(vec_any_ge,4,4) VEC_I_VI_VI(vec_any_ge,4,8)
  VEC_I_VU_VU(vec_any_ge,4,1) VEC_I_VU_VU(vec_any_ge,4,2) VEC_I_VU_VU(vec_any_ge,4,4) VEC_I_VU_VU(vec_any_ge,4,8)
  VEC_I_VR_VR(vec_any_ge,4,4) VEC_I_VR_VR(vec_any_ge,4,8)
  interface vec_any_ge
    procedure :: I_VI_VI(vec_any_ge,4,1), I_VI_VI(vec_any_ge,4,2), I_VI_VI(vec_any_ge,4,4), I_VI_VI(vec_any_ge,4,8)
    procedure :: I_VU_VU(vec_any_ge,4,1), I_VU_VU(vec_any_ge,4,2), I_VU_VU(vec_any_ge,4,4), I_VU_VU(vec_any_ge,4,8)
    procedure :: I_VR_VR(vec_any_ge,4,4), I_VR_VR(vec_any_ge,4,8)
  end interface vec_any_ge
  public :: vec_any_ge

#undef VEC_I_VR_VR
#undef VEC_I_VU_VU
#undef VEC_I_VI_VI
#undef I_VR_VR
#undef I_VU_VU
#undef I_VI_VI

!------------------------------------------
! vector function(vector, vector, integer)
!------------------------------------------
! i0 means the integer argument has ignore_tkr(k)
#define VI_VI_VI_I(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND##i0
#define VU_VU_VU_I(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##vu##VKIND##i0
#define VR_VR_VR_I(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##i0

#define VEC_VI_VI_VI_I(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##vi##VKIND##i) :: VI_VI_VI_I(NAME, VKIND);
#define VEC_VU_VU_VU_I(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND##vu##VKIND##i) :: VU_VU_VU_I(NAME, VKIND);
#define VEC_VR_VR_VR_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##i) :: VR_VR_VR_I(NAME, VKIND);

! vec_sld
  VEC_VI_VI_VI_I(vec_sld,1) VEC_VI_VI_VI_I(vec_sld,2) VEC_VI_VI_VI_I(vec_sld,4) VEC_VI_VI_VI_I(vec_sld,8)
  VEC_VU_VU_VU_I(vec_sld,1) VEC_VU_VU_VU_I(vec_sld,2) VEC_VU_VU_VU_I(vec_sld,4) VEC_VU_VU_VU_I(vec_sld,8)
  VEC_VR_VR_VR_I(vec_sld,4) VEC_VR_VR_VR_I(vec_sld,8)
  interface vec_sld
    procedure :: VI_VI_VI_I(vec_sld,1), VI_VI_VI_I(vec_sld,2), VI_VI_VI_I(vec_sld,4), VI_VI_VI_I(vec_sld,8)
    procedure :: VU_VU_VU_I(vec_sld,1), VU_VU_VU_I(vec_sld,2), VU_VU_VU_I(vec_sld,4), VU_VU_VU_I(vec_sld,8)
    procedure :: VR_VR_VR_I(vec_sld,4), VR_VR_VR_I(vec_sld,8)
  end interface vec_sld
  public :: vec_sld

! vec_sldw
  VEC_VI_VI_VI_I(vec_sldw,1) VEC_VI_VI_VI_I(vec_sldw,2) VEC_VI_VI_VI_I(vec_sldw,4) VEC_VI_VI_VI_I(vec_sldw,8)
  VEC_VU_VU_VU_I(vec_sldw,1) VEC_VU_VU_VU_I(vec_sldw,2) VEC_VU_VU_VU_I(vec_sldw,4) VEC_VU_VU_VU_I(vec_sldw,8)
  VEC_VR_VR_VR_I(vec_sldw,4) VEC_VR_VR_VR_I(vec_sldw,8)
  interface vec_sldw
    procedure :: VI_VI_VI_I(vec_sldw,1), VI_VI_VI_I(vec_sldw,2), VI_VI_VI_I(vec_sldw,4), VI_VI_VI_I(vec_sldw,8)
    procedure :: VU_VU_VU_I(vec_sldw,1), VU_VU_VU_I(vec_sldw,2), VU_VU_VU_I(vec_sldw,4), VU_VU_VU_I(vec_sldw,8)
    procedure :: VR_VR_VR_I(vec_sldw,4), VR_VR_VR_I(vec_sldw,8)
  end interface vec_sldw
  public :: vec_sldw

#undef VEC_VR_VR_VR_I
#undef VEC_VU_VU_VU_I
#undef VEC_VI_VI_VI_I
#undef VR_VR_VR_I
#undef VU_VU_VU_I
#undef VI_VI_VI_I

!----------------------------------
! vector function(vector, integer)
!----------------------------------
! 'i0' stands for the integer argument being ignored via
! the `ignore_tkr' directive.
#define VR_VI_I(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vi##VKIND##i0
#define VR_VU_I(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vu##VKIND##i0

#define VEC_VR_VI_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vi##VKIND##i) :: VR_VI_I(NAME, VKIND);
#define VEC_VR_VU_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vu##VKIND##i) :: VR_VU_I(NAME, VKIND);

! vec_ctf
  VEC_VR_VI_I(vec_ctf,4) VEC_VR_VI_I(vec_ctf,8)
  VEC_VR_VU_I(vec_ctf,4) VEC_VR_VU_I(vec_ctf,8)
  interface vec_ctf
     procedure :: VR_VI_I(vec_ctf,4), VR_VI_I(vec_ctf,8)
     procedure :: VR_VU_I(vec_ctf,4), VR_VU_I(vec_ctf,8)
  end interface vec_ctf
  public :: vec_ctf

#undef VEC_VR_VU_I
#undef VEC_VR_VI_I
#undef VR_VU_I
#undef VR_VI_I

end module __ppc_intrinsics
