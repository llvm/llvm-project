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

! vector(u) function f(vector(u))
#define ELEM_FUNC_VUVU(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vu##VKIND(arg1); \
    vector(unsigned(VKIND)), intent(in) :: arg1; \
  end function ;

! vector(r) function f(vector(r))
#define ELEM_FUNC_VRVR_2(VKIND1, VKIND2) \
  elemental vector(real(VKIND1)) function elem_func_vr##VKIND1##vr##VKIND2(arg1); \
    vector(real(VKIND2)), intent(in) :: arg1; \
  end function ;
#define ELEM_FUNC_VRVR(VKIND) ELEM_FUNC_VRVR_2(VKIND, VKIND)

! vector(i) function f(i)
#define ELEM_FUNC_VII_2(RKIND, VKIND) \
  elemental vector(integer(RKIND)) function elem_func_vi##RKIND##i##VKIND(arg1); \
    integer(VKIND), intent(in) :: arg1; \
  end function ;
#define ELEM_FUNC_VII(VKIND) ELEM_FUNC_VII_2(VKIND, VKIND)

! vector(r) function f(r)
#define ELEM_FUNC_VRR(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##r##VKIND(arg1); \
    real(VKIND), intent(in) :: arg1; \
  end function ;

  ELEM_FUNC_VIVI(1) ELEM_FUNC_VIVI(2) ELEM_FUNC_VIVI(4) ELEM_FUNC_VIVI(8)
  ELEM_FUNC_VUVU(1)
  ELEM_FUNC_VRVR_2(4,8) ELEM_FUNC_VRVR_2(8,4)
  ELEM_FUNC_VRVR(4) ELEM_FUNC_VRVR(8)
  ELEM_FUNC_VII_2(4,1) ELEM_FUNC_VII_2(4,2) ELEM_FUNC_VII_2(4,8)
  ELEM_FUNC_VII(1) ELEM_FUNC_VII(2) ELEM_FUNC_VII(4) ELEM_FUNC_VII(8)
  ELEM_FUNC_VRR(4) ELEM_FUNC_VRR(8)

#undef ELEM_FUNC_VRR
#undef ELEM_FUNC_VII
#undef ELEM_FUNC_VII_2
#undef ELEM_FUNC_VRVR
#undef ELEM_FUNC_VRVR_2
#undef ELEM_FUNC_VUVU
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

! integer function f(vector(i), i)
#define ELEM_FUNC_IVII(VKIND) \
  elemental integer(VKIND) function elem_func_i##VKIND##vi##VKIND##i(arg1, arg2); \
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

! real function f(vector(r), i)
#define ELEM_FUNC_RVRI(VKIND) \
  elemental real(VKIND) function elem_func_r##VKIND##vr##VKIND##i(arg1, arg2); \
    vector(real(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
  end function ;

! vector(i) function f(vector(i), i)
#define ELEM_FUNC_VIVII0(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND##i0(arg1, arg2); \
    vector(integer(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
  end function ;

! vector(u) function f(vector(u), i)
#define ELEM_FUNC_VUVUI0(VKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vu##VKIND##i0(arg1, arg2); \
    vector(unsigned(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
  end function ;

! vector(r) function f(vector(r), i)
#define ELEM_FUNC_VRVRI0(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##i0(arg1, arg2); \
    vector(real(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
  end function ;

! vector(i) function f(i, integer)
#define FUNC_VII0I(VKIND) \
  pure vector(integer(VKIND)) function func_vi##VKIND##i0i##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    integer(VKIND), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! vector(r) function f(i, real)
#define FUNC_VRI0R(VKIND) \
  pure vector(real(VKIND)) function func_vr##VKIND##i0r##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    real(VKIND), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! vector(i) function f(i, vector(i))
#define FUNC_VII0VI(VKIND) \
  pure vector(integer(VKIND)) function func_vi##VKIND##i0vi##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    vector(integer(VKIND)), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! vector(u) function f(i, vector(u))
#define FUNC_VUI0VU(VKIND) \
  pure vector(unsigned(VKIND)) function func_vu##VKIND##i0vu##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    vector(unsigned(VKIND)), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! vector(r) function f(i, vector(r))
#define FUNC_VRI0VR(VKIND) \
  pure vector(real(VKIND)) function func_vr##VKIND##i0vr##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    vector(real(VKIND)), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! vector(u(1)) function f(i, i)
#define FUNC_VU1I0I(KIND) \
  vector(unsigned(1)) function func_vu1i0i##KIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    integer(KIND), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! vector(u(1)) function f(i, r)
#define FUNC_VU1I0R(KIND) \
  vector(unsigned(1)) function func_vu1i0r##KIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    real(KIND), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function ;

! __vector_pair function f(i, vector(i))
#define FUNC_VPI0VI(VKIND) \
  pure __vector_pair function func_vpi0vi##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    vector(integer(VKIND)), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function;

! __vector_pair function f(i, vector(u))
#define FUNC_VPI0VU(VKIND) \
  pure __vector_pair function func_vpi0vu##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    vector(unsigned(VKIND)), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function;

! __vector_pair function f(i, vector(r))
#define FUNC_VPI0VR(VKIND) \
  pure __vector_pair function func_vpi0vr##VKIND(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    vector(real(VKIND)), intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function;

! __vector_pair function f(i, __vector_pair)
#define FUNC_VPI0VP \
  pure __vector_pair function func_vpi0vp(arg1, arg2); \
    integer(8), intent(in) :: arg1; \
    !dir$ ignore_tkr(k) arg1; \
    __vector_pair, intent(in) :: arg2; \
    !dir$ ignore_tkr(r) arg2; \
  end function;

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
  ELEM_FUNC_IVII(1) ELEM_FUNC_IVII(2) ELEM_FUNC_IVII(4) ELEM_FUNC_IVII(8)
  ELEM_FUNC_RVRI(4) ELEM_FUNC_RVRI(8)
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
  ELEM_FUNC_VIVII0(1) ELEM_FUNC_VIVII0(2) ELEM_FUNC_VIVII0(4) ELEM_FUNC_VIVII0(8)
  ELEM_FUNC_VUVUI0(1) ELEM_FUNC_VUVUI0(2) ELEM_FUNC_VUVUI0(4) ELEM_FUNC_VUVUI0(8)
  ELEM_FUNC_VRVRI0(4) ELEM_FUNC_VRVRI0(8)
  FUNC_VII0VI(1) FUNC_VII0VI(2) FUNC_VII0VI(4) FUNC_VII0VI(8)
  FUNC_VUI0VU(1) FUNC_VUI0VU(2) FUNC_VUI0VU(4) FUNC_VUI0VU(8)
  FUNC_VRI0VR(4) FUNC_VRI0VR(8)
  FUNC_VII0I(1) FUNC_VII0I(2) FUNC_VII0I(4) FUNC_VII0I(8)
  FUNC_VRI0R(4) FUNC_VRI0R(8)
  FUNC_VPI0VI(1) FUNC_VPI0VI(2) FUNC_VPI0VI(4) FUNC_VPI0VI(8)
  FUNC_VPI0VU(1) FUNC_VPI0VU(2) FUNC_VPI0VU(4) FUNC_VPI0VU(8)
  FUNC_VPI0VR(4) FUNC_VPI0VR(8)
  FUNC_VPI0VP
  FUNC_VU1I0I(1) FUNC_VU1I0I(2) FUNC_VU1I0I(4)
  FUNC_VU1I0R(4)

#undef FUNC_VEC_CONVERT_VRVIVR
#undef FUNC_VEC_CONVERT_VUVIVU
#undef FUNC_VEC_CONVERT_VIVIVI
#undef FUNC_VPI0VP
#undef FUNC_VPI0VR
#undef FUNC_VPI0VU
#undef FUNC_VPI0VI
#undef FUNC_VU1I0R
#undef FUNC_VU1I0I
#undef FUNC_VRI0VR
#undef FUNC_VUI0VU
#undef FUNC_VII0VI
#undef FUNC_VRI0R
#undef FUNC_VII0I
#undef ELEM_FUNC_VRVRI0
#undef ELEM_FUNC_VUVUI0
#undef ELEM_FUNC_VIVII0
#undef ELEM_FUNC_RVRI
#undef ELEM_FUNC_VRVUI
#undef ELEM_FUNC_IVII
#undef ELEM_FUNC_VRVII
#undef ELEM_FUNC_IVRVR
#undef ELEM_FUNC_IVUVU
#undef ELEM_FUNC_IVIVI
#undef ELEM_FUNC_VUVRVR
#undef ELEM_FUNC_VRVRVU_2
#undef ELEM_FUNC_VRVRVR
#undef ELEM_FUNC_VUVUVU
#undef ELEM_FUNC_VUVUVU_2
#undef ELEM_FUNC_VIVIVU
#undef ELEM_FUNC_VIVIVU_2
#undef ELEM_FUNC_VUVIVI
#undef ELEM_FUNC_VIVIVI

!! ================ 3 arguments function interface ================
! vector(r) function f(vector(r), vector(r), vector(r))
#define ELEM_FUNC_VRVRVRVR(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1, arg2, arg3; \
  end function ;

! vector(i) function f(vector(i), vector(i), vector(u))
#define ELEM_FUNC_VIVIVIVU_2(VKIND, UKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##vi##VKIND##vi##VKIND##vu##UKIND(arg1, arg2, arg3); \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
    vector(unsigned(UKIND)), intent(in) :: arg3; \
  end function ;
#define ELEM_FUNC_VIVIVIVU(VKIND) ELEM_FUNC_VIVIVIVU_2(VKIND, VKIND)

! vector(u) function f(vector(u), vector(u), vector(u))
#define ELEM_FUNC_VUVUVUVU_2(VKIND, UKIND) \
  elemental vector(unsigned(VKIND)) function elem_func_vu##VKIND##vu##VKIND##vu##VKIND##vu##UKIND(arg1, arg2, arg3); \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2; \
    vector(unsigned(UKIND)), intent(in) :: arg3; \
  end function ;
#define ELEM_FUNC_VUVUVUVU(VKIND) ELEM_FUNC_VUVUVUVU_2(VKIND, VKIND)
  
! vector(r) function f(vector(r), vector(r), vector(u))
#define ELEM_FUNC_VRVRVRVU_2(VKIND, UKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vu##UKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
    vector(unsigned(UKIND)), intent(in) :: arg3; \
  end function ;
#define ELEM_FUNC_VRVRVRVU(VKIND) ELEM_FUNC_VRVRVRVU_2(VKIND, VKIND)

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

! vector(i) function f(i, vector(i), i)
#define ELEM_FUNC_VIIVII(VKIND) \
  elemental vector(integer(VKIND)) function elem_func_vi##VKIND##i##VKIND##vi##VKIND##i(arg1, arg2, arg3); \
    integer(VKIND), intent(in) :: arg1; \
    vector(integer(VKIND)), intent(in) :: arg2; \
    integer(8), intent(in) :: arg3; \
    !dir$ ignore_tkr(k) arg3; \
  end function ;

! vector(r) function f(r, vector(r), i)
#define ELEM_FUNC_VRRVRI(VKIND) \
  elemental vector(real(VKIND)) function elem_func_vr##VKIND##r##VKIND##vr##VKIND##i(arg1, arg2, arg3); \
    real(VKIND), intent(in) :: arg1; \
    vector(real(VKIND)), intent(in) :: arg2; \
    integer(8), intent(in) :: arg3; \
    !dir$ ignore_tkr(k) arg3; \
  end function ;

  ELEM_FUNC_VIVIVIVU(1) ELEM_FUNC_VIVIVIVU(2) ELEM_FUNC_VIVIVIVU(4) ELEM_FUNC_VIVIVIVU(8)
  ELEM_FUNC_VUVUVUVU(1) ELEM_FUNC_VUVUVUVU(2) ELEM_FUNC_VUVUVUVU(4) ELEM_FUNC_VUVUVUVU(8)
  ELEM_FUNC_VRVRVRVU(4) ELEM_FUNC_VRVRVRVU(8)
  ELEM_FUNC_VIVIVIVU_2(2,1) ELEM_FUNC_VIVIVIVU_2(4,1) ELEM_FUNC_VIVIVIVU_2(8,1)
  ELEM_FUNC_VUVUVUVU_2(2,1) ELEM_FUNC_VUVUVUVU_2(4,1) ELEM_FUNC_VUVUVUVU_2(8,1)
  ELEM_FUNC_VRVRVRVU_2(4,1) ELEM_FUNC_VRVRVRVU_2(8,1)
  ELEM_FUNC_VIIVII(1) ELEM_FUNC_VIIVII(2) ELEM_FUNC_VIIVII(4) ELEM_FUNC_VIIVII(8)
  ELEM_FUNC_VRRVRI(4) ELEM_FUNC_VRRVRI(8)
  ELEM_FUNC_VRVRVRVR(4) ELEM_FUNC_VRVRVRVR(8)
  ELEM_FUNC_VIVIVII(1) ELEM_FUNC_VIVIVII(2) ELEM_FUNC_VIVIVII(4) ELEM_FUNC_VIVIVII(8)
  ELEM_FUNC_VUVUVUI(1) ELEM_FUNC_VUVUVUI(2) ELEM_FUNC_VUVUVUI(4) ELEM_FUNC_VUVUVUI(8)
  ELEM_FUNC_VRVRVRI(4) ELEM_FUNC_VRVRVRI(8)

#undef ELEM_FUNC_VRRVRI
#undef ELEM_FUNC_VIIVII
#undef ELEM_FUNC_VRVRVRI
#undef ELEM_FUNC_VUVUVUI
#undef ELEM_FUNC_VIVIVII
#undef ELEM_FUNC_VRVRVRVU
#undef ELEM_FUNC_VRVRVRVU_2
#undef ELEM_FUNC_VUVUVUVU
#undef ELEM_FUNC_VUVUVUVU_2
#undef ELEM_FUNC_VIVIVIVU
#undef ELEM_FUNC_VIVIVIVU_2
#undef ELEM_FUNC_VRVRVRVR

!! ================ 3 argument subroutine interfaces =================================
! subroutine(vector(i), i, vector(i))
#define SUB_VIIVI(VKIND) \
  pure subroutine sub_vi##VKIND##ivi##VKIND(arg1, arg2, arg3); \
    vector(integer(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    vector(integer(VKIND)), intent(in) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine ;

! subroutine(vector(u), i, vector(u))
#define SUB_VUIVU(VKIND) \
  pure subroutine sub_vu##VKIND##ivu##VKIND(arg1, arg2, arg3); \
    vector(unsigned(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    vector(unsigned(VKIND)), intent(in) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine ;

! subroutine(vector(r), i, vector(r))
#define SUB_VRIVR(VKIND) \
  pure subroutine sub_vr##VKIND##ivr##VKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    vector(real(VKIND)), intent(in) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine ;

! subroutine(vector(i), i, i)
#define SUB_VIII(VKIND) \
  pure subroutine sub_vi##VKIND##ii##VKIND(arg1, arg2, arg3); \
    vector(integer(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    integer(VKIND), intent(out) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine ;

! subroutine(vector(u), i, i)
#define SUB_VUII(VKIND) \
  pure subroutine sub_vu##VKIND##ii##VKIND(arg1, arg2, arg3); \
    vector(unsigned(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    integer(VKIND), intent(out) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine ;

! subroutine(vector(r), i, r)
#define SUB_VRIR(VKIND) \
  pure subroutine sub_vr##VKIND##ir##VKIND(arg1, arg2, arg3); \
    vector(real(VKIND)), intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    real(VKIND), intent(out) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine ;

! subroutine(__vector_pair, i, __vector_pair)
  pure subroutine sub_vpi0vp(arg1, arg2, arg3)
    __vector_pair, intent(in) :: arg1
    integer(8), intent(in) :: arg2
    !dir$ ignore_tkr(k) arg2
    __vector_pair, intent(out) :: arg3
    !dir$ ignore_tkr(r) arg3
  end subroutine

! subroutine(__vector_pair, i, vector(i))
#define SUB_VPI0VI(VKIND) \
  pure subroutine sub_vpi0vi##VKIND(arg1, arg2, arg3); \
    __vector_pair, intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    vector(integer(VKIND)), intent(out) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine;

! subroutine(__vector_pair, i, vector(u))
#define SUB_VPI0VU(VKIND) \
  pure subroutine sub_vpi0vu##VKIND(arg1, arg2, arg3); \
    __vector_pair, intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    vector(unsigned(VKIND)), intent(out) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine;

! subroutine(__vector_pair, i, vector(r))
#define SUB_VPI0VR(VKIND) \
  pure subroutine sub_vpi0vr##VKIND(arg1, arg2, arg3); \
    __vector_pair, intent(in) :: arg1; \
    integer(8), intent(in) :: arg2; \
    !dir$ ignore_tkr(k) arg2; \
    vector(real(VKIND)), intent(out) :: arg3; \
    !dir$ ignore_tkr(r) arg3; \
  end subroutine;

! subroutine(__vector_pair, i, i)
  pure subroutine sub_vpi0i0(arg1, arg2, arg3)
    __vector_pair, intent(in) :: arg1
    integer(8), intent(in) :: arg2
    !dir$ ignore_tkr(k) arg2
    integer(8), intent(out) :: arg3
    !dir$ ignore_tkr(kr) arg3
  end subroutine

! subroutine(__vector_pair, i, r)
  pure subroutine sub_vpi0r0(arg1, arg2, arg3)
    __vector_pair, intent(in) :: arg1
    integer(8), intent(in) :: arg2
    !dir$ ignore_tkr(k) arg2
    real(8), intent(out) :: arg3
    !dir$ ignore_tkr(kr) arg3
  end subroutine

  SUB_VIIVI(1) SUB_VIIVI(2) SUB_VIIVI(4) SUB_VIIVI(8)
  SUB_VUIVU(1) SUB_VUIVU(2) SUB_VUIVU(4) SUB_VUIVU(8)
  SUB_VRIVR(4) SUB_VRIVR(8)
  SUB_VIII(1) SUB_VIII(2) SUB_VIII(4) SUB_VIII(8)
  SUB_VUII(1) SUB_VUII(2) SUB_VUII(4) SUB_VUII(8)
  SUB_VRIR(4) SUB_VRIR(8)
  SUB_VPI0VI(1) SUB_VPI0VI(2) SUB_VPI0VI(4) SUB_VPI0VI(8)
  SUB_VPI0VU(1) SUB_VPI0VU(2) SUB_VPI0VU(4) SUB_VPI0VU(8)
  SUB_VPI0VR(4) SUB_VPI0VR(8)

#undef SUB_VPI0VR
#undef SUB_VPI0VU
#undef SUB_VPI0VI
#undef SUB_VRIR
#undef SUB_VUII
#undef SUB_VIII
#undef SUB_VRIVR
#undef SUB_VUIVU
#undef SUB_VIIVI

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

!-----------------------------
! vector function(vector/i/r)
!-----------------------------
#define VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND
#define VU_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND
#define VR_VR_2(NAME, VKIND1, VKIND2) __ppc_##NAME##_vr##VKIND1##vr##VKIND2
#define VR_VR(NAME, VKIND) VR_VR_2(NAME, VKIND, VKIND)
#define VI_I_2(NAME, RKIND, VKIND) __ppc_##NAME##_vi##RKIND##i##VKIND
#define VI_I(NAME, VKIND) VI_I_2(NAME, VKIND, VKIND)
#define VR_R(NAME, VKIND) __ppc_##NAME##_vr##VKIND##r##VKIND

#define VEC_VI_VI(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND) :: VI_VI(NAME, VKIND);
#define VEC_VU_VU(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND) :: VU_VU(NAME, VKIND);
#define VEC_VR_VR_2(NAME, VKIND1, VKIND2) \
  procedure(elem_func_vr##VKIND1##vr##VKIND2) :: VR_VR_2(NAME, VKIND1, VKIND2);
#define VEC_VR_VR(NAME, VKIND) VEC_VR_VR_2(NAME, VKIND, VKIND)
#define VEC_VI_I_2(NAME, RKIND, VKIND) \
  procedure(elem_func_vi##RKIND##i##VKIND) :: VI_I_2(NAME, RKIND, VKIND);
#define VEC_VI_I(NAME, VKIND) VEC_VI_I_2(NAME, VKIND, VKIND)
#define VEC_VR_R(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##r##VKIND) :: VR_R(NAME, VKIND);

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

! vec_cvbf16spn
  VEC_VU_VU(vec_cvbf16spn,1)
  interface vec_cvbf16spn
    procedure :: VU_VU(vec_cvbf16spn,1)
  end interface
  public vec_cvbf16spn

! vec_cvspbf16
  VEC_VU_VU(vec_cvspbf16_,1)
  interface vec_cvspbf16
    procedure :: VU_VU(vec_cvspbf16_,1)
  end interface
  public vec_cvspbf16

! vec_splats
  VEC_VI_I(vec_splats,1) VEC_VI_I(vec_splats,2) VEC_VI_I(vec_splats,4) VEC_VI_I(vec_splats,8)
  VEC_VR_R(vec_splats,4) VEC_VR_R(vec_splats,8)
  interface vec_splats
     procedure :: VI_I(vec_splats,1), VI_I(vec_splats,2), VI_I(vec_splats,4), VI_I(vec_splats,8)
     procedure :: VR_R(vec_splats,4), VR_R(vec_splats,8)
  end interface vec_splats
  public :: vec_splats

! vec_splat_32
  VEC_VI_I_2(vec_splat_s32_,4,1) VEC_VI_I_2(vec_splat_s32_,4,2) VEC_VI_I_2(vec_splat_s32_,4,4) VEC_VI_I_2(vec_splat_s32_,4,8)
  interface vec_splat_s32
     procedure :: VI_I_2(vec_splat_s32_,4,1), VI_I_2(vec_splat_s32_,4,2), VI_I_2(vec_splat_s32_,4,4), VI_I_2(vec_splat_s32_,4,8)
  end interface vec_splat_s32
  public :: vec_splat_s32

#undef VEC_VR_R
#undef VEC_VI_I
#undef VEC_VI_I_2
#undef VEC_VR_VR
#undef VEC_VR_VR_2
#undef VEC_VU_VU
#undef VEC_VI_VI
#undef VR_R
#undef VI_I
#undef VI_I_2
#undef VR_VR
#undef VR_VR_2
#undef VU_VU
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

! vec_mergeh
  VEC_VI_VI_VI(vec_mergeh,1) VEC_VI_VI_VI(vec_mergeh,2) VEC_VI_VI_VI(vec_mergeh,4) VEC_VI_VI_VI(vec_mergeh,8)
  VEC_VU_VU_VU(vec_mergeh,1) VEC_VU_VU_VU(vec_mergeh,2) VEC_VU_VU_VU(vec_mergeh,4) VEC_VU_VU_VU(vec_mergeh,8)
  VEC_VR_VR_VR(vec_mergeh,4) VEC_VR_VR_VR(vec_mergeh,8)
  interface vec_mergeh
    procedure :: VI_VI_VI(vec_mergeh,1), VI_VI_VI(vec_mergeh,2), VI_VI_VI(vec_mergeh,4), VI_VI_VI(vec_mergeh,8)
    procedure :: VU_VU_VU(vec_mergeh,1), VU_VU_VU(vec_mergeh,2), VU_VU_VU(vec_mergeh,4), VU_VU_VU(vec_mergeh,8)
    procedure :: VR_VR_VR(vec_mergeh,4), VR_VR_VR(vec_mergeh,8)
  end interface vec_mergeh
  public :: vec_mergeh

! vec_mergel
  VEC_VI_VI_VI(vec_mergel,1) VEC_VI_VI_VI(vec_mergel,2) VEC_VI_VI_VI(vec_mergel,4) VEC_VI_VI_VI(vec_mergel,8)
  VEC_VU_VU_VU(vec_mergel,1) VEC_VU_VU_VU(vec_mergel,2) VEC_VU_VU_VU(vec_mergel,4) VEC_VU_VU_VU(vec_mergel,8)
  VEC_VR_VR_VR(vec_mergel,4) VEC_VR_VR_VR(vec_mergel,8)
  interface vec_mergel
    procedure :: VI_VI_VI(vec_mergel,1), VI_VI_VI(vec_mergel,2), VI_VI_VI(vec_mergel,4), VI_VI_VI(vec_mergel,8)
    procedure :: VU_VU_VU(vec_mergel,1), VU_VU_VU(vec_mergel,2), VU_VU_VU(vec_mergel,4), VU_VU_VU(vec_mergel,8)
    procedure :: VR_VR_VR(vec_mergel,4), VR_VR_VR(vec_mergel,8)
  end interface vec_mergel
  public :: vec_mergel

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

!-------------------------------------------------------
! vector(unsigned(1)) function(integer, i/r)
!-------------------------------------------------------
#define VU1_I0_I(NAME, KIND) __ppc_##NAME##_vu1i0i##KIND
#define VU1_I0_R(NAME, KIND) __ppc_##NAME##_vu1i0r##KIND

#define VEC_VU1_I0_I(NAME, KIND) \
  procedure(func_vu1i0i##KIND) :: VU1_I0_I(NAME, KIND);
#define VEC_VU1_I0_R(NAME, KIND) \
  procedure(func_vu1i0r##KIND) :: VU1_I0_R(NAME, KIND);

! vec_lvsl
  VEC_VU1_I0_I(vec_lvsl,1) VEC_VU1_I0_I(vec_lvsl,2) VEC_VU1_I0_I(vec_lvsl,4)
  VEC_VU1_I0_R(vec_lvsl,4)
  interface vec_lvsl
    procedure :: VU1_I0_I(vec_lvsl,1), VU1_I0_I(vec_lvsl,2), VU1_I0_I(vec_lvsl,4)
    procedure :: VU1_I0_R(vec_lvsl,4)
  end interface
  public :: vec_lvsl

! vec_lvsr
  VEC_VU1_I0_I(vec_lvsr,1) VEC_VU1_I0_I(vec_lvsr,2) VEC_VU1_I0_I(vec_lvsr,4)
  VEC_VU1_I0_R(vec_lvsr,4)
  interface vec_lvsr
    procedure :: VU1_I0_I(vec_lvsr,1), VU1_I0_I(vec_lvsr,2), VU1_I0_I(vec_lvsr,4)
    procedure :: VU1_I0_R(vec_lvsr,4)
  end interface
  public :: vec_lvsr

#undef VEC_VU1_I0_R
#undef VEC_VU1_I0_I
#undef VU1_I0_R
#undef VU1_I0_I

!-------------------------------------------------------
! vector function(integer, i/u/r/vector)
!-------------------------------------------------------
! i0 means the integer argument has ignore_tkr(k)
#define VI_I0_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##i0##vi##VKIND
#define VU_I0_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##i0##vu##VKIND
#define VR_I0_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##i0##vr##VKIND
#define VI_I0_I(NAME, VKIND) __ppc_##NAME##_vi##VKIND##i0##i##VKIND
#define VR_I0_R(NAME, VKIND) __ppc_##NAME##_vr##VKIND##i0##r##VKIND

#define VEC_VI_I0_VI(NAME, VKIND) \
  procedure(func_vi##VKIND##i0##vi##VKIND) :: VI_I0_VI(NAME, VKIND);
#define VEC_VU_I0_VU(NAME, VKIND) \
  procedure(func_vu##VKIND##i0##vu##VKIND) :: VU_I0_VU(NAME, VKIND);
#define VEC_VR_I0_VR(NAME, VKIND) \
  procedure(func_vr##VKIND##i0##vr##VKIND) :: VR_I0_VR(NAME, VKIND);
#define VEC_VI_I0_I(NAME, VKIND) \
  procedure(func_vi##VKIND##i0##i##VKIND) :: VI_I0_I(NAME, VKIND);
#define VEC_VR_I0_R(NAME, VKIND) \
  procedure(func_vr##VKIND##i0##r##VKIND) :: VR_I0_R(NAME, VKIND);

! vec_ld
  VEC_VI_I0_VI(vec_ld,1) VEC_VI_I0_VI(vec_ld,2) VEC_VI_I0_VI(vec_ld,4)
  VEC_VU_I0_VU(vec_ld,1) VEC_VU_I0_VU(vec_ld,2) VEC_VU_I0_VU(vec_ld,4)
  VEC_VR_I0_VR(vec_ld,4)
  VEC_VI_I0_I(vec_ld,1) VEC_VI_I0_I(vec_ld,2) VEC_VI_I0_I(vec_ld,4)
  VEC_VR_I0_R(vec_ld,4)
  interface vec_ld
    procedure :: VI_I0_VI(vec_ld,1), VI_I0_VI(vec_ld,2), VI_I0_VI(vec_ld,4)
    procedure :: VU_I0_VU(vec_ld,1), VU_I0_VU(vec_ld,2), VU_I0_VU(vec_ld,4)
    procedure :: VR_I0_VR(vec_ld,4)
    procedure :: VI_I0_I(vec_ld,1), VI_I0_I(vec_ld,2), VI_I0_I(vec_ld,4)
    procedure :: VR_I0_R(vec_ld,4)
  end interface
  public :: vec_ld

! vec_lde
  VEC_VI_I0_I(vec_lde,1) VEC_VI_I0_I(vec_lde,2) VEC_VI_I0_I(vec_lde,4)
  VEC_VR_I0_R(vec_lde,4)
  interface vec_lde
    procedure :: VI_I0_I(vec_lde,1), VI_I0_I(vec_lde,2), VI_I0_I(vec_lde,4)
    procedure :: VR_I0_R(vec_lde,4)
  end interface
  public :: vec_lde

! vec_ldl
  VEC_VI_I0_VI(vec_ldl,1) VEC_VI_I0_VI(vec_ldl,2) VEC_VI_I0_VI(vec_ldl,4)
  VEC_VU_I0_VU(vec_ldl,1) VEC_VU_I0_VU(vec_ldl,2) VEC_VU_I0_VU(vec_ldl,4)
  VEC_VR_I0_VR(vec_ldl,4)
  VEC_VI_I0_I(vec_ldl,1) VEC_VI_I0_I(vec_ldl,2) VEC_VI_I0_I(vec_ldl,4)
  VEC_VR_I0_R(vec_ldl,4)
  interface vec_ldl
    procedure :: VI_I0_VI(vec_ldl,1), VI_I0_VI(vec_ldl,2), VI_I0_VI(vec_ldl,4)
    procedure :: VU_I0_VU(vec_ldl,1), VU_I0_VU(vec_ldl,2), VU_I0_VU(vec_ldl,4)
    procedure :: VR_I0_VR(vec_ldl,4)
    procedure :: VI_I0_I(vec_ldl,1), VI_I0_I(vec_ldl,2), VI_I0_I(vec_ldl,4)
    procedure :: VR_I0_R(vec_ldl,4)
  end interface
  public :: vec_ldl

! vec_lxv
  VEC_VI_I0_VI(vec_lxv,1) VEC_VI_I0_VI(vec_lxv,2) VEC_VI_I0_VI(vec_lxv,4) VEC_VI_I0_VI(vec_lxv,8)
  VEC_VU_I0_VU(vec_lxv,1) VEC_VU_I0_VU(vec_lxv,2) VEC_VU_I0_VU(vec_lxv,4) VEC_VU_I0_VU(vec_lxv,8)
  VEC_VR_I0_VR(vec_lxv,4) VEC_VR_I0_VR(vec_lxv,8)
  VEC_VI_I0_I(vec_lxv,1) VEC_VI_I0_I(vec_lxv,2) VEC_VI_I0_I(vec_lxv,4) VEC_VI_I0_I(vec_lxv,8)
  VEC_VR_I0_R(vec_lxv,4) VEC_VR_I0_R(vec_lxv,8)
  interface vec_lxv
    procedure :: VI_I0_VI(vec_lxv,1), VI_I0_VI(vec_lxv,2), VI_I0_VI(vec_lxv,4), VI_I0_VI(vec_lxv,8)
    procedure :: VU_I0_VU(vec_lxv,1), VU_I0_VU(vec_lxv,2), VU_I0_VU(vec_lxv,4), VU_I0_VU(vec_lxv,8)
    procedure :: VR_I0_VR(vec_lxv,4), VR_I0_VR(vec_lxv,8)
    procedure :: VI_I0_I(vec_lxv,1), VI_I0_I(vec_lxv,2), VI_I0_I(vec_lxv,4), VI_I0_I(vec_lxv,8)
    procedure :: VR_I0_R(vec_lxv,4), VR_I0_R(vec_lxv,8)
  end interface
  public :: vec_lxv

! vec_xl
  VEC_VI_I0_VI(vec_xl,1) VEC_VI_I0_VI(vec_xl,2) VEC_VI_I0_VI(vec_xl,4) VEC_VI_I0_VI(vec_xl,8)
  VEC_VU_I0_VU(vec_xl,1) VEC_VU_I0_VU(vec_xl,2) VEC_VU_I0_VU(vec_xl,4) VEC_VU_I0_VU(vec_xl,8)
  VEC_VR_I0_VR(vec_xl,4) VEC_VR_I0_VR(vec_xl,8)
  VEC_VI_I0_I(vec_xl,1) VEC_VI_I0_I(vec_xl,2) VEC_VI_I0_I(vec_xl,4) VEC_VI_I0_I(vec_xl,8)
  VEC_VR_I0_R(vec_xl,4) VEC_VR_I0_R(vec_xl,8)
  interface vec_xl
    procedure :: VI_I0_VI(vec_xl,1), VI_I0_VI(vec_xl,2), VI_I0_VI(vec_xl,4), VI_I0_VI(vec_xl,8)
    procedure :: VU_I0_VU(vec_xl,1), VU_I0_VU(vec_xl,2), VU_I0_VU(vec_xl,4), VU_I0_VU(vec_xl,8)
    procedure :: VR_I0_VR(vec_xl,4), VR_I0_VR(vec_xl,8)
    procedure :: VI_I0_I(vec_xl,1), VI_I0_I(vec_xl,2), VI_I0_I(vec_xl,4), VI_I0_I(vec_xl,8)
    procedure :: VR_I0_R(vec_xl,4), VR_I0_R(vec_xl,8)
  end interface
  public :: vec_xl

! vec_xl_be
  VEC_VI_I0_VI(vec_xl_be,1) VEC_VI_I0_VI(vec_xl_be,2) VEC_VI_I0_VI(vec_xl_be,4) VEC_VI_I0_VI(vec_xl_be,8)
  VEC_VU_I0_VU(vec_xl_be,1) VEC_VU_I0_VU(vec_xl_be,2) VEC_VU_I0_VU(vec_xl_be,4) VEC_VU_I0_VU(vec_xl_be,8)
  VEC_VR_I0_VR(vec_xl_be,4) VEC_VR_I0_VR(vec_xl_be,8)
  VEC_VI_I0_I(vec_xl_be,1) VEC_VI_I0_I(vec_xl_be,2) VEC_VI_I0_I(vec_xl_be,4) VEC_VI_I0_I(vec_xl_be,8)
  VEC_VR_I0_R(vec_xl_be,4) VEC_VR_I0_R(vec_xl_be,8)
  interface vec_xl_be
    procedure :: VI_I0_VI(vec_xl_be,1), VI_I0_VI(vec_xl_be,2), VI_I0_VI(vec_xl_be,4), VI_I0_VI(vec_xl_be,8)
    procedure :: VU_I0_VU(vec_xl_be,1), VU_I0_VU(vec_xl_be,2), VU_I0_VU(vec_xl_be,4), VU_I0_VU(vec_xl_be,8)
    procedure :: VR_I0_VR(vec_xl_be,4), VR_I0_VR(vec_xl_be,8)
    procedure :: VI_I0_I(vec_xl_be,1), VI_I0_I(vec_xl_be,2), VI_I0_I(vec_xl_be,4) , VI_I0_I(vec_xl_be,8)
    procedure :: VR_I0_R(vec_xl_be,4), VR_I0_R(vec_xl_be,8)
  end interface
  public :: vec_xl_be

! vec_xld2
  VEC_VI_I0_VI(vec_xld2_,1) VEC_VI_I0_VI(vec_xld2_,2) VEC_VI_I0_VI(vec_xld2_,4) VEC_VI_I0_VI(vec_xld2_,8)
  VEC_VU_I0_VU(vec_xld2_,1) VEC_VU_I0_VU(vec_xld2_,2) VEC_VU_I0_VU(vec_xld2_,4) VEC_VU_I0_VU(vec_xld2_,8)
  VEC_VR_I0_VR(vec_xld2_,4) VEC_VR_I0_VR(vec_xld2_,8)
  VEC_VI_I0_I(vec_xld2_,1) VEC_VI_I0_I(vec_xld2_,2) VEC_VI_I0_I(vec_xld2_,4) VEC_VI_I0_I(vec_xld2_,8)
  VEC_VR_I0_R(vec_xld2_,4) VEC_VR_I0_R(vec_xld2_,8)
  interface vec_xld2
    procedure :: VI_I0_VI(vec_xld2_,1), VI_I0_VI(vec_xld2_,2), VI_I0_VI(vec_xld2_,4), VI_I0_VI(vec_xld2_,8)
    procedure :: VU_I0_VU(vec_xld2_,1), VU_I0_VU(vec_xld2_,2), VU_I0_VU(vec_xld2_,4), VU_I0_VU(vec_xld2_,8)
    procedure :: VR_I0_VR(vec_xld2_,4), VR_I0_VR(vec_xld2_,8)
    procedure :: VI_I0_I(vec_xld2_,1), VI_I0_I(vec_xld2_,2), VI_I0_I(vec_xld2_,4), VI_I0_I(vec_xld2_,8)
    procedure :: VR_I0_R(vec_xld2_,4), VR_I0_R(vec_xld2_,8)
  end interface
  public :: vec_xld2

! vec_xlds
  VEC_VI_I0_VI(vec_xlds,8)
  VEC_VU_I0_VU(vec_xlds,8)
  VEC_VR_I0_VR(vec_xlds,8)
  VEC_VI_I0_I(vec_xlds,8)
  VEC_VR_I0_R(vec_xlds,8)
  interface vec_xlds
    procedure :: VI_I0_VI(vec_xlds,8)
    procedure :: VU_I0_VU(vec_xlds,8)
    procedure :: VR_I0_VR(vec_xlds,8)
    procedure :: VI_I0_I(vec_xlds,8)
    procedure :: VR_I0_R(vec_xlds,8)
  end interface
  public :: vec_xlds

! vec_xlw4
  VEC_VI_I0_VI(vec_xlw4_,1) VEC_VI_I0_VI(vec_xlw4_,2)
  VEC_VU_I0_VU(vec_xlw4_,1) VEC_VU_I0_VU(vec_xlw4_,2) VEC_VU_I0_VU(vec_xlw4_,4)
  VEC_VR_I0_VR(vec_xlw4_,4)
  VEC_VI_I0_I(vec_xlw4_,1) VEC_VI_I0_I(vec_xlw4_,2) VEC_VI_I0_I(vec_xlw4_,4)
  VEC_VR_I0_R(vec_xlw4_,4)
  interface vec_xlw4
    procedure :: VI_I0_VI(vec_xlw4_,1), VI_I0_VI(vec_xlw4_,2)
    procedure :: VU_I0_VU(vec_xlw4_,1), VU_I0_VU(vec_xlw4_,2), VU_I0_VU(vec_xlw4_,4)
    procedure :: VR_I0_VR(vec_xlw4_,4)
    procedure :: VI_I0_I(vec_xlw4_,1), VI_I0_I(vec_xlw4_,2), VI_I0_I(vec_xlw4_,4)
    procedure :: VR_I0_R(vec_xlw4_,4)
  end interface
  public :: vec_xlw4

#undef VEC_VR_I0_R
#undef VEC_VI_I0_I
#undef VEC_VR_I0_VR
#undef VEC_VU_I0_VU
#undef VEC_VI_I0_VI
#undef VR_I0_R
#undef VI_I0_I
#undef VR_I0_VR
#undef VU_I0_VU
#undef VI_I0_VI

!-------------------------------------------------------
! __vector_pair function(integer, vector/__vector_pair)
!-------------------------------------------------------
#define VP_I0_VI(NAME, VKIND) __ppc_##NAME##_vpi0##vi##VKIND
#define VP_I0_VU(NAME, VKIND) __ppc_##NAME##_vpi0##vu##VKIND
#define VP_I0_VR(NAME, VKIND) __ppc_##NAME##_vpi0##vr##VKIND
#define VP_I0_VP(NAME) __ppc_##NAME##_vpi0vp0

#define VEC_VP_I0_VI(NAME, VKIND) \
  procedure(func_vpi0vi##VKIND) :: VP_I0_VI(NAME, VKIND);
#define VEC_VP_I0_VU(NAME, VKIND) \
  procedure(func_vpi0vu##VKIND) :: VP_I0_VU(NAME, VKIND);
#define VEC_VP_I0_VR(NAME, VKIND) \
  procedure(func_vpi0vr##VKIND) :: VP_I0_VR(NAME, VKIND);
#define VEC_VP_I0_VP(NAME) procedure(func_vpi0vp) :: VP_I0_VP(NAME);

! vec_lxvp
  VEC_VP_I0_VI(vec_lxvp,1) VEC_VP_I0_VI(vec_lxvp,2) VEC_VP_I0_VI(vec_lxvp,4) VEC_VP_I0_VI(vec_lxvp,8)
  VEC_VP_I0_VU(vec_lxvp,1) VEC_VP_I0_VU(vec_lxvp,2) VEC_VP_I0_VU(vec_lxvp,4) VEC_VP_I0_VU(vec_lxvp,8)
  VEC_VP_I0_VR(vec_lxvp,4) VEC_VP_I0_VR(vec_lxvp,8)
  VEC_VP_I0_VP(vec_lxvp)
  interface vec_lxvp
     procedure :: VP_I0_VI(vec_lxvp,1), VP_I0_VI(vec_lxvp,2), VP_I0_VI(vec_lxvp,4), VP_I0_VI(vec_lxvp,8)
     procedure :: VP_I0_VU(vec_lxvp,1), VP_I0_VU(vec_lxvp,2), VP_I0_VU(vec_lxvp,4), VP_I0_VU(vec_lxvp,8)
     procedure :: VP_I0_VR(vec_lxvp,4), VP_I0_VR(vec_lxvp,8)
     procedure :: VP_I0_VP(vec_lxvp)
  end interface vec_lxvp
  public :: vec_lxvp

! vsx_lxvp (alias to vec_lxvp)
  interface vsx_lxvp
     procedure :: VP_I0_VI(vec_lxvp,1), VP_I0_VI(vec_lxvp,2), VP_I0_VI(vec_lxvp,4), VP_I0_VI(vec_lxvp,8)
     procedure :: VP_I0_VU(vec_lxvp,1), VP_I0_VU(vec_lxvp,2), VP_I0_VU(vec_lxvp,4), VP_I0_VU(vec_lxvp,8)
     procedure :: VP_I0_VR(vec_lxvp,4), VP_I0_VR(vec_lxvp,8)
     procedure :: VP_I0_VP(vec_lxvp)
  end interface vsx_lxvp
  public :: vsx_lxvp

#undef VEC_VP_I0_VP
#undef VEC_VP_I0_VR
#undef VEC_VP_I0_VU
#undef VEC_VP_I0_VI
#undef VP_I0_VP
#undef VP_I0_VR
#undef VP_I0_VU
#undef VP_I0_VI

!-----------------------------------------
! vector function(vector, vector, vector)
!-----------------------------------------
#define VR_VR_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND
#define VI_VI_VI_VU_2(NAME, VKIND, UKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND##vu##UKIND
#define VI_VI_VI_VU(NAME, VKIND) VI_VI_VI_VU_2(NAME, VKIND, VKIND)
#define VU_VU_VU_VU_2(NAME, VKIND, UKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##vu##VKIND##vu##UKIND
#define VU_VU_VU_VU(NAME, VKIND) VU_VU_VU_VU_2(NAME, VKIND, VKIND)
#define VR_VR_VR_VU_2(NAME, VKIND, UKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##vu##UKIND
#define VR_VR_VR_VU(NAME, VKIND) VR_VR_VR_VU_2(NAME, VKIND, VKIND)
! i0 indicates "!dir$ ignore_tkr(k) arg3"
#define VI_VI_VI_I(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND##i0
#define VU_VU_VU_I(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##vu##VKIND##i0
#define VR_VR_VR_I(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##i0

#define VEC_VR_VR_VR_VR(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND) :: VR_VR_VR_VR(NAME, VKIND);
#define VEC_VI_VI_VI_VU_2(NAME, VKIND, UKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##vi##VKIND##vu##UKIND) :: VI_VI_VI_VU_2(NAME, VKIND, UKIND);
#define VEC_VI_VI_VI_VU(NAME, VKIND) VEC_VI_VI_VI_VU_2(NAME, VKIND, VKIND)
#define VEC_VU_VU_VU_VU_2(NAME, VKIND, UKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND##vu##VKIND##vu##UKIND) :: VU_VU_VU_VU_2(NAME, VKIND, UKIND);
#define VEC_VU_VU_VU_VU(NAME, VKIND) VEC_VU_VU_VU_VU_2(NAME, VKIND, VKIND)
#define VEC_VR_VR_VR_VU_2(NAME, VKIND, UKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##vu##UKIND) :: VR_VR_VR_VU_2(NAME, VKIND, UKIND);
#define VEC_VR_VR_VR_VU(NAME, VKIND) VEC_VR_VR_VR_VU_2(NAME, VKIND, VKIND)
#define VEC_VI_VI_VI_I(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##vi##VKIND##i) :: VI_VI_VI_I(NAME, VKIND);
#define VEC_VU_VU_VU_I(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND##vu##VKIND##i) :: VU_VU_VU_I(NAME, VKIND);
#define VEC_VR_VR_VR_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##vr##VKIND##i) :: VR_VR_VR_I(NAME, VKIND);

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

! vec_perm
  VEC_VI_VI_VI_VU_2(vec_perm,1,1) VEC_VI_VI_VI_VU_2(vec_perm,2,1) VEC_VI_VI_VI_VU_2(vec_perm,4,1) VEC_VI_VI_VI_VU_2(vec_perm,8,1)
  VEC_VU_VU_VU_VU_2(vec_perm,1,1) VEC_VU_VU_VU_VU_2(vec_perm,2,1) VEC_VU_VU_VU_VU_2(vec_perm,4,1) VEC_VU_VU_VU_VU_2(vec_perm,8,1)
  VEC_VR_VR_VR_VU_2(vec_perm,4,1) VEC_VR_VR_VR_VU_2(vec_perm,8,1)
  interface vec_perm
     procedure :: VI_VI_VI_VU_2(vec_perm,1,1), VI_VI_VI_VU_2(vec_perm,2,1), VI_VI_VI_VU_2(vec_perm,4,1), VI_VI_VI_VU_2(vec_perm,8,1)
     procedure :: VU_VU_VU_VU_2(vec_perm,1,1), VU_VU_VU_VU_2(vec_perm,2,1), VU_VU_VU_VU_2(vec_perm,4,1), VU_VU_VU_VU_2(vec_perm,8,1)
     procedure :: VR_VR_VR_VU_2(vec_perm,4,1), VR_VR_VR_VU_2(vec_perm,8,1)
  end interface vec_perm
  public :: vec_perm

! vec_permi
  VEC_VI_VI_VI_I(vec_permi,8)
  VEC_VU_VU_VU_I(vec_permi,8)
  VEC_VR_VR_VR_I(vec_permi,4) VEC_VR_VR_VR_I(vec_permi,8)
  interface vec_permi
     procedure :: VI_VI_VI_I(vec_permi,8)
     procedure :: VU_VU_VU_I(vec_permi,8)
     procedure :: VR_VR_VR_I(vec_permi,4), VR_VR_VR_I(vec_permi,8)
  end interface vec_permi
  public :: vec_permi

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

#undef VEC_VR_VR_VR_I
#undef VEC_VU_VU_VU_I
#undef VEC_VI_VI_VI_I
#undef VEC_VI_VI_VI_VU_2
#undef VEC_VI_VI_VI_VU
#undef VEC_VU_VU_VU_VU_2
#undef VEC_VU_VU_VU_VU
#undef VEC_VR_VR_VR_VU_2
#undef VEC_VR_VR_VR_VU
#undef VEC_VR_VR_VR_VR
#undef VR_VR_VR_I
#undef VU_VU_VU_I
#undef VI_VI_VI_I
#undef VI_VI_VI_VU
#undef VI_VI_VI_VU_2
#undef VU_VU_VU_VU
#undef VU_VU_VU_VU_2
#undef VR_VR_VR_VU
#undef VR_VR_VR_VU_2
#undef VR_VR_VR_VR

!------------------------------------------
! vector function(integer, vector, integer)
! vector function(real, vector, integer)
!------------------------------------------
#define VI_I_VI_I(NAME, VKIND) __ppc_##NAME##_vi##VKIND##i##VKIND##vi##VKIND##i0
#define VR_R_VR_I(NAME, VKIND) __ppc_##NAME##_vr##VKIND##r##VKIND##vr##VKIND##i0

#define VEC_VI_I_VI_I(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##i##VKIND##vi##VKIND##i) :: VI_I_VI_I(NAME, VKIND);
#define VEC_VR_R_VR_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##r##VKIND##vr##VKIND##i) :: VR_R_VR_I(NAME, VKIND);

! vec_insert
  VEC_VI_I_VI_I(vec_insert,1) VEC_VI_I_VI_I(vec_insert,2) VEC_VI_I_VI_I(vec_insert,4) VEC_VI_I_VI_I(vec_insert,8)
  VEC_VR_R_VR_I(vec_insert,4) VEC_VR_R_VR_I(vec_insert,8)
  interface vec_insert
     procedure :: VI_I_VI_I(vec_insert,1), VI_I_VI_I(vec_insert,2), VI_I_VI_I(vec_insert,4), VI_I_VI_I(vec_insert,8)
     procedure :: VR_R_VR_I(vec_insert,4), VR_R_VR_I(vec_insert,8)
  end interface vec_insert
  public :: vec_insert

#undef VEC_VR_R_VR_I
#undef VEC_VI_I_VI_I
#undef VR_R_VR_I
#undef VI_I_VI_I

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

!----------------------------------------
! integer/real function(vector, integer)
!----------------------------------------
#define I_VI_I(NAME, VKIND) __ppc_##NAME##_i##VKIND##vi##VKIND##i0
#define R_VR_I(NAME, VKIND) __ppc_##NAME##_r##VKIND##vr##VKIND##i0

#define VEC_I_VI_I(NAME, VKIND) \
  procedure(elem_func_i##VKIND##vi##VKIND##i) :: I_VI_I(NAME, VKIND);
#define VEC_R_VR_I(NAME, VKIND) \
  procedure(elem_func_r##VKIND##vr##VKIND##i) :: R_VR_I(NAME, VKIND);

! vec_extract
  VEC_I_VI_I(vec_extract,1) VEC_I_VI_I(vec_extract,2) VEC_I_VI_I(vec_extract,4) VEC_I_VI_I(vec_extract,8)
  VEC_R_VR_I(vec_extract,4) VEC_R_VR_I(vec_extract,8)
  interface vec_extract
     procedure :: I_VI_I(vec_extract,1), I_VI_I(vec_extract,2), I_VI_I(vec_extract,4), I_VI_I(vec_extract,8)
     procedure :: R_VR_I(vec_extract,4), R_VR_I(vec_extract,8)
  end interface
  public :: vec_extract

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
#define VI_VI_I0(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##i0
#define VU_VU_I0(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##i0
#define VR_VR_I0(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##i0

#define VEC_VR_VI_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vi##VKIND##i) :: VR_VI_I(NAME, VKIND);
#define VEC_VR_VU_I(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vu##VKIND##i) :: VR_VU_I(NAME, VKIND);
#define VEC_VI_VI_I0(NAME, VKIND) \
  procedure(elem_func_vi##VKIND##vi##VKIND##i0) :: VI_VI_I0(NAME, VKIND);
#define VEC_VU_VU_I0(NAME, VKIND) \
  procedure(elem_func_vu##VKIND##vu##VKIND##i0) :: VU_VU_I0(NAME, VKIND);
#define VEC_VR_VR_I0(NAME, VKIND) \
  procedure(elem_func_vr##VKIND##vr##VKIND##i0) :: VR_VR_I0(NAME, VKIND);

! vec_ctf
  VEC_VR_VI_I(vec_ctf,4) VEC_VR_VI_I(vec_ctf,8)
  VEC_VR_VU_I(vec_ctf,4) VEC_VR_VU_I(vec_ctf,8)
  interface vec_ctf
     procedure :: VR_VI_I(vec_ctf,4), VR_VI_I(vec_ctf,8)
     procedure :: VR_VU_I(vec_ctf,4), VR_VU_I(vec_ctf,8)
  end interface vec_ctf
  public :: vec_ctf

! vec_splat
  VEC_VI_VI_I0(vec_splat,1) VEC_VI_VI_I0(vec_splat,2) VEC_VI_VI_I0(vec_splat,4) VEC_VI_VI_I0(vec_splat,8)
  VEC_VU_VU_I0(vec_splat,1) VEC_VU_VU_I0(vec_splat,2) VEC_VU_VU_I0(vec_splat,4) VEC_VU_VU_I0(vec_splat,8)
  VEC_VR_VR_I0(vec_splat,4) VEC_VR_VR_I0(vec_splat,8)
  interface vec_splat
     procedure :: VI_VI_I0(vec_splat,1), VI_VI_I0(vec_splat,2), VI_VI_I0(vec_splat,4), VI_VI_I0(vec_splat,8)
     procedure :: VU_VU_I0(vec_splat,1), VU_VU_I0(vec_splat,2), VU_VU_I0(vec_splat,4), VU_VU_I0(vec_splat,8)
     procedure :: VR_VR_I0(vec_splat,4), VR_VR_I0(vec_splat,8)
  end interface vec_splat
  public :: vec_splat

#undef VEC_VR_VR_I0
#undef VEC_VU_VU_I0
#undef VEC_VI_VI_I0
#undef VEC_VR_VU_I
#undef VEC_VR_VI_I
#undef VR_VR_I0
#undef VU_VU_I0
#undef VI_VI_I0
#undef VR_VU_I
#undef VR_VI_I

!--------------------------------------------------
! subroutine(vector, integer, vector/integer/real)
!--------------------------------------------------
! 'i0' stands for the integer argument being ignored via
! the `ignore_tkr' directive.
#define SUB_VI_I_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##i0vi##VKIND
#define SUB_VU_I_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##i0vu##VKIND
#define SUB_VR_I_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##i0vr##VKIND
#define SUB_VI_I_I(NAME, VKIND) __ppc_##NAME##_vi##VKIND##i0i##VKIND
#define SUB_VU_I_I(NAME, VKIND) __ppc_##NAME##_vu##VKIND##i0u##VKIND
#define SUB_VR_I_R(NAME, VKIND) __ppc_##NAME##_vr##VKIND##i0r##VKIND

#define VEC_SUB_VI_I_VI(NAME, VKIND) \
  procedure(sub_vi##VKIND##ivi##VKIND) :: SUB_VI_I_VI(NAME, VKIND);
#define VEC_SUB_VU_I_VU(NAME, VKIND) \
  procedure(sub_vu##VKIND##ivu##VKIND) :: SUB_VU_I_VU(NAME, VKIND);
#define VEC_SUB_VR_I_VR(NAME, VKIND) \
  procedure(sub_vr##VKIND##ivr##VKIND) :: SUB_VR_I_VR(NAME, VKIND);
#define VEC_SUB_VI_I_I(NAME, VKIND) \
  procedure(sub_vi##VKIND##ii##VKIND) :: SUB_VI_I_I(NAME, VKIND);
#define VEC_SUB_VU_I_I(NAME, VKIND) \
  procedure(sub_vu##VKIND##ii##VKIND) :: SUB_VU_I_I(NAME, VKIND);
#define VEC_SUB_VR_I_R(NAME, VKIND) \
  procedure(sub_vr##VKIND##ir##VKIND) :: SUB_VR_I_R(NAME, VKIND);

! vec_st
  VEC_SUB_VI_I_VI(vec_st,1) VEC_SUB_VI_I_VI(vec_st,2) VEC_SUB_VI_I_VI(vec_st,4)
  VEC_SUB_VU_I_VU(vec_st,1) VEC_SUB_VU_I_VU(vec_st,2) VEC_SUB_VU_I_VU(vec_st,4)
  VEC_SUB_VR_I_VR(vec_st,4)
  VEC_SUB_VI_I_I(vec_st,1) VEC_SUB_VI_I_I(vec_st,2) VEC_SUB_VI_I_I(vec_st,4)
  VEC_SUB_VU_I_I(vec_st,1) VEC_SUB_VU_I_I(vec_st,2) VEC_SUB_VU_I_I(vec_st,4)
  VEC_SUB_VR_I_R(vec_st,4)
  interface vec_st
    procedure :: SUB_VI_I_VI(vec_st,1), SUB_VI_I_VI(vec_st,2), SUB_VI_I_VI(vec_st,4)
    procedure :: SUB_VU_I_VU(vec_st,1), SUB_VU_I_VU(vec_st,2), SUB_VU_I_VU(vec_st,4)
    procedure :: SUB_VR_I_VR(vec_st,4)
    procedure :: SUB_VI_I_I(vec_st,1), SUB_VI_I_I(vec_st,2), SUB_VI_I_I(vec_st,4)
    procedure :: SUB_VU_I_I(vec_st,1), SUB_VU_I_I(vec_st,2), SUB_VU_I_I(vec_st,4)
    procedure :: SUB_VR_I_R(vec_st,4)
  end interface vec_st
  public :: vec_st

! vec_ste
  VEC_SUB_VI_I_I(vec_ste,1) VEC_SUB_VI_I_I(vec_ste,2) VEC_SUB_VI_I_I(vec_ste,4)
  VEC_SUB_VU_I_I(vec_ste,1) VEC_SUB_VU_I_I(vec_ste,2) VEC_SUB_VU_I_I(vec_ste,4)
  VEC_SUB_VR_I_R(vec_ste,4)
  interface vec_ste
    procedure :: SUB_VI_I_I(vec_ste,1), SUB_VI_I_I(vec_ste,2), SUB_VI_I_I(vec_ste,4)
    procedure :: SUB_VU_I_I(vec_ste,1), SUB_VU_I_I(vec_ste,2), SUB_VU_I_I(vec_ste,4)
    procedure :: SUB_VR_I_R(vec_ste,4)
  end interface vec_ste
  public :: vec_ste

! vec_stxv
  VEC_SUB_VI_I_VI(vec_stxv,1) VEC_SUB_VI_I_VI(vec_stxv,2) VEC_SUB_VI_I_VI(vec_stxv,4) VEC_SUB_VI_I_VI(vec_stxv,8)
  VEC_SUB_VU_I_VU(vec_stxv,1) VEC_SUB_VU_I_VU(vec_stxv,2) VEC_SUB_VU_I_VU(vec_stxv,4) VEC_SUB_VU_I_VU(vec_stxv,8)
  VEC_SUB_VR_I_VR(vec_stxv,4) VEC_SUB_VR_I_VR(vec_stxv,8)
  VEC_SUB_VI_I_I(vec_stxv,1) VEC_SUB_VI_I_I(vec_stxv,2) VEC_SUB_VI_I_I(vec_stxv,4) VEC_SUB_VI_I_I(vec_stxv,8)
  VEC_SUB_VU_I_I(vec_stxv,1) VEC_SUB_VU_I_I(vec_stxv,2) VEC_SUB_VU_I_I(vec_stxv,4) VEC_SUB_VU_I_I(vec_stxv,8)
  VEC_SUB_VR_I_R(vec_stxv,4) VEC_SUB_VR_I_R(vec_stxv,8)
  interface vec_stxv
    procedure :: SUB_VI_I_VI(vec_stxv,1), SUB_VI_I_VI(vec_stxv,2), SUB_VI_I_VI(vec_stxv,4), SUB_VI_I_VI(vec_stxv,8)
    procedure :: SUB_VU_I_VU(vec_stxv,1), SUB_VU_I_VU(vec_stxv,2), SUB_VU_I_VU(vec_stxv,4), SUB_VU_I_VU(vec_stxv,8)
    procedure :: SUB_VR_I_VR(vec_stxv,4), SUB_VR_I_VR(vec_stxv,8)
    procedure :: SUB_VI_I_I(vec_stxv,1), SUB_VI_I_I(vec_stxv,2), SUB_VI_I_I(vec_stxv,4), SUB_VI_I_I(vec_stxv,8)
    procedure :: SUB_VU_I_I(vec_stxv,1), SUB_VU_I_I(vec_stxv,2), SUB_VU_I_I(vec_stxv,4), SUB_VU_I_I(vec_stxv,8)
    procedure :: SUB_VR_I_R(vec_stxv,4), SUB_VR_I_R(vec_stxv,8)
  end interface vec_stxv
  public :: vec_stxv

! vec_xst
  VEC_SUB_VI_I_VI(vec_xst,1) VEC_SUB_VI_I_VI(vec_xst,2) VEC_SUB_VI_I_VI(vec_xst,4) VEC_SUB_VI_I_VI(vec_xst,8)
  VEC_SUB_VU_I_VU(vec_xst,1) VEC_SUB_VU_I_VU(vec_xst,2) VEC_SUB_VU_I_VU(vec_xst,4) VEC_SUB_VU_I_VU(vec_xst,8)
  VEC_SUB_VR_I_VR(vec_xst,4) VEC_SUB_VR_I_VR(vec_xst,8)
  VEC_SUB_VI_I_I(vec_xst,1) VEC_SUB_VI_I_I(vec_xst,2) VEC_SUB_VI_I_I(vec_xst,4) VEC_SUB_VI_I_I(vec_xst,8)
  VEC_SUB_VU_I_I(vec_xst,1) VEC_SUB_VU_I_I(vec_xst,2) VEC_SUB_VU_I_I(vec_xst,4) VEC_SUB_VU_I_I(vec_xst,8)
  VEC_SUB_VR_I_R(vec_xst,4) VEC_SUB_VR_I_R(vec_xst,8)
  interface vec_xst
    procedure :: SUB_VI_I_VI(vec_xst,1), SUB_VI_I_VI(vec_xst,2), SUB_VI_I_VI(vec_xst,4), SUB_VI_I_VI(vec_xst,8)
    procedure :: SUB_VU_I_VU(vec_xst,1), SUB_VU_I_VU(vec_xst,2), SUB_VU_I_VU(vec_xst,4), SUB_VU_I_VU(vec_xst,8)
    procedure :: SUB_VR_I_VR(vec_xst,4), SUB_VR_I_VR(vec_xst,8)
    procedure :: SUB_VI_I_I(vec_xst,1), SUB_VI_I_I(vec_xst,2), SUB_VI_I_I(vec_xst,4), SUB_VI_I_I(vec_xst,8)
    procedure :: SUB_VU_I_I(vec_xst,1), SUB_VU_I_I(vec_xst,2), SUB_VU_I_I(vec_xst,4), SUB_VU_I_I(vec_xst,8)
    procedure :: SUB_VR_I_R(vec_xst,4), SUB_VR_I_R(vec_xst,8)
  end interface vec_xst
  public :: vec_xst

! vec_xst_be
  VEC_SUB_VI_I_VI(vec_xst_be,1) VEC_SUB_VI_I_VI(vec_xst_be,2) VEC_SUB_VI_I_VI(vec_xst_be,4) VEC_SUB_VI_I_VI(vec_xst_be,8)
  VEC_SUB_VU_I_VU(vec_xst_be,1) VEC_SUB_VU_I_VU(vec_xst_be,2) VEC_SUB_VU_I_VU(vec_xst_be,4) VEC_SUB_VU_I_VU(vec_xst_be,8)
  VEC_SUB_VR_I_VR(vec_xst_be,4) VEC_SUB_VR_I_VR(vec_xst_be,8)
  VEC_SUB_VI_I_I(vec_xst_be,1) VEC_SUB_VI_I_I(vec_xst_be,2) VEC_SUB_VI_I_I(vec_xst_be,4) VEC_SUB_VI_I_I(vec_xst_be,8)
  VEC_SUB_VU_I_I(vec_xst_be,1) VEC_SUB_VU_I_I(vec_xst_be,2) VEC_SUB_VU_I_I(vec_xst_be,4) VEC_SUB_VU_I_I(vec_xst_be,8)
  VEC_SUB_VR_I_R(vec_xst_be,4) VEC_SUB_VR_I_R(vec_xst_be,8)
  interface vec_xst_be
    procedure :: SUB_VI_I_VI(vec_xst_be,1), SUB_VI_I_VI(vec_xst_be,2), SUB_VI_I_VI(vec_xst_be,4), SUB_VI_I_VI(vec_xst_be,8)
    procedure :: SUB_VU_I_VU(vec_xst_be,1), SUB_VU_I_VU(vec_xst_be,2), SUB_VU_I_VU(vec_xst_be,4), SUB_VU_I_VU(vec_xst_be,8)
    procedure :: SUB_VR_I_VR(vec_xst_be,4), SUB_VR_I_VR(vec_xst_be,8)
    procedure :: SUB_VI_I_I(vec_xst_be,1), SUB_VI_I_I(vec_xst_be,2), SUB_VI_I_I(vec_xst_be,4), SUB_VI_I_I(vec_xst_be,8)
    procedure :: SUB_VU_I_I(vec_xst_be,1), SUB_VU_I_I(vec_xst_be,2), SUB_VU_I_I(vec_xst_be,4), SUB_VU_I_I(vec_xst_be,8)
    procedure :: SUB_VR_I_R(vec_xst_be,4), SUB_VR_I_R(vec_xst_be,8)
  end interface vec_xst_be
  public :: vec_xst_be

! vec_xstd2
  VEC_SUB_VI_I_VI(vec_xstd2_,1) VEC_SUB_VI_I_VI(vec_xstd2_,2) VEC_SUB_VI_I_VI(vec_xstd2_,4) VEC_SUB_VI_I_VI(vec_xstd2_,8)
  VEC_SUB_VU_I_VU(vec_xstd2_,1) VEC_SUB_VU_I_VU(vec_xstd2_,2) VEC_SUB_VU_I_VU(vec_xstd2_,4) VEC_SUB_VU_I_VU(vec_xstd2_,8)
  VEC_SUB_VR_I_VR(vec_xstd2_,4) VEC_SUB_VR_I_VR(vec_xstd2_,8)
  VEC_SUB_VI_I_I(vec_xstd2_,1) VEC_SUB_VI_I_I(vec_xstd2_,2) VEC_SUB_VI_I_I(vec_xstd2_,4) VEC_SUB_VI_I_I(vec_xstd2_,8)
  VEC_SUB_VU_I_I(vec_xstd2_,1) VEC_SUB_VU_I_I(vec_xstd2_,2) VEC_SUB_VU_I_I(vec_xstd2_,4) VEC_SUB_VU_I_I(vec_xstd2_,8)
  VEC_SUB_VR_I_R(vec_xstd2_,4) VEC_SUB_VR_I_R(vec_xstd2_,8)
  interface vec_xstd2
    procedure :: SUB_VI_I_VI(vec_xstd2_,1), SUB_VI_I_VI(vec_xstd2_,2), SUB_VI_I_VI(vec_xstd2_,4), SUB_VI_I_VI(vec_xstd2_,8)
    procedure :: SUB_VU_I_VU(vec_xstd2_,1), SUB_VU_I_VU(vec_xstd2_,2), SUB_VU_I_VU(vec_xstd2_,4), SUB_VU_I_VU(vec_xstd2_,8)
    procedure :: SUB_VR_I_VR(vec_xstd2_,4), SUB_VR_I_VR(vec_xstd2_,8)
    procedure :: SUB_VI_I_I(vec_xstd2_,1), SUB_VI_I_I(vec_xstd2_,2), SUB_VI_I_I(vec_xstd2_,4), SUB_VI_I_I(vec_xstd2_,8)
    procedure :: SUB_VU_I_I(vec_xstd2_,1), SUB_VU_I_I(vec_xstd2_,2), SUB_VU_I_I(vec_xstd2_,4), SUB_VU_I_I(vec_xstd2_,8)
    procedure :: SUB_VR_I_R(vec_xstd2_,4), SUB_VR_I_R(vec_xstd2_,8)
  end interface vec_xstd2
  public :: vec_xstd2

! vec_xstw4
  VEC_SUB_VI_I_VI(vec_xstw4_,1) VEC_SUB_VI_I_VI(vec_xstw4_,2) VEC_SUB_VI_I_VI(vec_xstw4_,4)
  VEC_SUB_VU_I_VU(vec_xstw4_,1) VEC_SUB_VU_I_VU(vec_xstw4_,2) VEC_SUB_VU_I_VU(vec_xstw4_,4)
  VEC_SUB_VR_I_VR(vec_xstw4_,4)
  VEC_SUB_VI_I_I(vec_xstw4_,1) VEC_SUB_VI_I_I(vec_xstw4_,2) VEC_SUB_VI_I_I(vec_xstw4_,4)
  VEC_SUB_VU_I_I(vec_xstw4_,1) VEC_SUB_VU_I_I(vec_xstw4_,2) VEC_SUB_VU_I_I(vec_xstw4_,4)
  VEC_SUB_VR_I_R(vec_xstw4_,4)
  interface vec_xstw4
    procedure :: SUB_VI_I_VI(vec_xstw4_,1), SUB_VI_I_VI(vec_xstw4_,2), SUB_VI_I_VI(vec_xstw4_,4)
    procedure :: SUB_VU_I_VU(vec_xstw4_,1), SUB_VU_I_VU(vec_xstw4_,2), SUB_VU_I_VU(vec_xstw4_,4)
    procedure :: SUB_VR_I_VR(vec_xstw4_,4)
    procedure :: SUB_VI_I_I(vec_xstw4_,1), SUB_VI_I_I(vec_xstw4_,2), SUB_VI_I_I(vec_xstw4_,4)
    procedure :: SUB_VU_I_I(vec_xstw4_,1), SUB_VU_I_I(vec_xstw4_,2), SUB_VU_I_I(vec_xstw4_,4)
    procedure :: SUB_VR_I_R(vec_xstw4_,4)
  end interface vec_xstw4
  public :: vec_xstw4

#undef VEC_SUB_VI_I_VI
#undef VEC_SUB_VU_I_VU
#undef VEC_SUB_VR_I_VR
#undef VEC_SUB_VI_I_I
#undef VEC_SUB_VU_I_I
#undef VEC_SUB_VR_I_R
#undef SUB_VI_I_VI
#undef SUB_VU_I_VU
#undef SUB_VR_I_VR
#undef SUB_VI_I_I
#undef SUB_VU_I_I
#undef SUB_VR_Ik_R

!-----------------------------------------------------------------------
! subroutine(__vector_pair, integer, __vector_pair/vector/integer/real)
!-----------------------------------------------------------------------
#define VP_I0_VI(NAME, VKIND) __ppc_##NAME##_vpi0vi##VKIND
#define VP_I0_VU(NAME, VKIND) __ppc_##NAME##_vpi0vu##VKIND
#define VP_I0_VR(NAME, VKIND) __ppc_##NAME##_vpi0vr##VKIND

#define VEC_VP_I0_VI(NAME, VKIND) \
  procedure(sub_vpi0vi##VKIND) :: VP_I0_VI(NAME, VKIND);
#define VEC_VP_I0_VU(NAME, VKIND) \
  procedure(sub_vpi0vu##VKIND) :: VP_I0_VU(NAME, VKIND);
#define VEC_VP_I0_VR(NAME, VKIND) \
  procedure(sub_vpi0vr##VKIND) :: VP_I0_VR(NAME, VKIND);

! vec_stxvp
  procedure(sub_vpi0vp) :: __ppc_vec_stxvp_vpi0vp0
  procedure(sub_vpi0i0) :: __ppc_vec_stxvp_vpi0i0
  procedure(sub_vpi0r0) :: __ppc_vec_stxvp_vpi0r0
  VEC_VP_I0_VI(vec_stxvp, 1) VEC_VP_I0_VI(vec_stxvp, 2) VEC_VP_I0_VI(vec_stxvp, 4) VEC_VP_I0_VI(vec_stxvp, 8)
  VEC_VP_I0_VU(vec_stxvp, 1) VEC_VP_I0_VU(vec_stxvp, 2) VEC_VP_I0_VU(vec_stxvp, 4) VEC_VP_I0_VU(vec_stxvp, 8)
  VEC_VP_I0_VR(vec_stxvp, 4) VEC_VP_I0_VR(vec_stxvp, 8)
  interface vec_stxvp
     procedure :: __ppc_vec_stxvp_vpi0vp0
     procedure :: __ppc_vec_stxvp_vpi0i0
     procedure :: __ppc_vec_stxvp_vpi0r0
     procedure :: VP_I0_VI(vec_stxvp, 1), VP_I0_VI(vec_stxvp, 2), VP_I0_VI(vec_stxvp, 4), VP_I0_VI(vec_stxvp, 8)
     procedure :: VP_I0_VU(vec_stxvp, 1), VP_I0_VU(vec_stxvp, 2), VP_I0_VU(vec_stxvp, 4), VP_I0_VU(vec_stxvp, 8)
     procedure :: VP_I0_VR(vec_stxvp, 4), VP_I0_VR(vec_stxvp, 8)
  end interface vec_stxvp
  public :: vec_stxvp

! vsx_stxvp (alias to vec_stxvp)
  interface vsx_stxvp
     procedure :: __ppc_vec_stxvp_vpi0vp0
     procedure :: __ppc_vec_stxvp_vpi0i0
     procedure :: __ppc_vec_stxvp_vpi0r0
     procedure :: VP_I0_VI(vec_stxvp, 1), VP_I0_VI(vec_stxvp, 2), VP_I0_VI(vec_stxvp, 4), VP_I0_VI(vec_stxvp, 8)
     procedure :: VP_I0_VU(vec_stxvp, 1), VP_I0_VU(vec_stxvp, 2), VP_I0_VU(vec_stxvp, 4), VP_I0_VU(vec_stxvp, 8)
     procedure :: VP_I0_VR(vec_stxvp, 4), VP_I0_VR(vec_stxvp, 8)
  end interface vsx_stxvp
  public :: vsx_stxvp

#undef VEC_VP_I0_VR
#undef VEC_VP_I0_VU
#undef VEC_VP_I0_VI
#undef VP_I0_VR
#undef VP_I0_VU
#undef VP_I0_VI

end module __ppc_intrinsics
