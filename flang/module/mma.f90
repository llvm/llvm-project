!===-- module/mma.f90 ------------------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

module mma
  implicit none
  private

  abstract interface

!! ========== 1 argument subroutine interface ================================!!
!! subroutine s(__vector_quad)
  elemental subroutine sub_vq(acc)
    __vector_quad, intent(inout) :: acc
  end subroutine

!! ========== 2 argument subroutine interface ================================!!
!! __vector_pair function f(i, vector(i))
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

  FUNC_VPI0VI(1) FUNC_VPI0VI(2) FUNC_VPI0VI(4) FUNC_VPI0VI(8)
  FUNC_VPI0VU(1) FUNC_VPI0VU(2) FUNC_VPI0VU(4) FUNC_VPI0VU(8)
  FUNC_VPI0VR(4) FUNC_VPI0VR(8)
  FUNC_VPI0VP

#undef FUNC_VPI0VP
#undef FUNC_VPI0VR
#undef FUNC_VPI0VU 
#undef FUNC_VPI0VI 

!! ========== 3 arguments subroutine interface ===============================!!
!! __vector_pair subroutine s(vp, integer, vector(i))
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

! subroutine(__vector_pair, i, __vector_pair)
  pure subroutine sub_vpi0vp(arg1, arg2, arg3)
    __vector_pair, intent(in) :: arg1
    integer(8), intent(in) :: arg2
    !dir$ ignore_tkr(k) arg2
    __vector_pair, intent(out) :: arg3
    !dir$ ignore_tkr(r) arg3
  end subroutine

!! subroutine s(__vector_pair, vector(i), vector(i))
#define ELEM_SUB_VPVIVI(VKIND) \
  elemental subroutine sub_vpvi##VKIND##vi##VKIND(pair, arg1, arg2); \
    __vector_pair, intent(out) :: pair ; \
    vector(integer(VKIND)), intent(in) :: arg1, arg2; \
  end subroutine ;

!! subroutine s(__vector_pair, vector(u), vector(u))
#define ELEM_SUB_VPVUVU(VKIND) \
  elemental subroutine sub_vpvu##VKIND##vu##VKIND(pair, arg1, arg2); \
    __vector_pair, intent(out) :: pair ; \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2; \
  end subroutine ;

!! subroutine s(__vector_pair, vector(r), vector(r))
#define ELEM_SUB_VPVRVR(VKIND) \
  elemental subroutine sub_vpvr##VKIND##vr##VKIND(pair, arg1, arg2); \
    __vector_pair, intent(out) :: pair ; \
    vector(real(VKIND)), intent(in) :: arg1, arg2; \
  end subroutine ;

  ELEM_SUB_VPVIVI(1) ELEM_SUB_VPVIVI(2)
  ELEM_SUB_VPVIVI(4) ELEM_SUB_VPVIVI(8)
  ELEM_SUB_VPVUVU(1) ELEM_SUB_VPVUVU(2)
  ELEM_SUB_VPVUVU(4) ELEM_SUB_VPVUVU(8)
  ELEM_SUB_VPVRVR(4) ELEM_SUB_VPVRVR(8)
  SUB_VPI0VI(1) SUB_VPI0VI(2) SUB_VPI0VI(4) SUB_VPI0VI(8)
  SUB_VPI0VU(1) SUB_VPI0VU(2) SUB_VPI0VU(4) SUB_VPI0VU(8)
  SUB_VPI0VR(4) SUB_VPI0VR(8)

#undef ELEM_SUB_VPVIVI
#undef ELEM_SUB_VPVUVU
#undef ELEM_SUB_VPVRVR
#undef SUB_VPI0VR
#undef SUB_VPI0VU
#undef SUB_VPI0VI

!! subroutine s(__vector_quad, vector(i), vector(i))
#define ELEM_SUB_VQVIVI(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vi##VKIND##vi##VKIND(acc, a, b); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(integer(VKIND)), intent(in) :: a, b; \
  end subroutine ;

!! subroutine s(__vector_quad, vector(u), vector(u))
#define ELEM_SUB_VQVUVU(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vu##VKIND##vu##VKIND(acc, a, b); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(unsigned(VKIND)), intent(in) :: a, b; \
  end subroutine ;

!! subroutine s(__vector_quad, vector(r), vector(r))
#define ELEM_SUB_VQVRVR(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vr##VKIND##vr##VKIND(acc, a, b); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(real(VKIND)), intent(in) :: a, b; \
  end subroutine ;

  ELEM_SUB_VQVIVI(inout,1) ELEM_SUB_VQVIVI(inout,2)
  ELEM_SUB_VQVUVU(inout,1)
  ELEM_SUB_VQVRVR(inout,4)
  ELEM_SUB_VQVIVI(out,1) ELEM_SUB_VQVIVI(out,2)
  ELEM_SUB_VQVUVU(out,1)
  ELEM_SUB_VQVRVR(out,4)

#undef ELEM_SUB_VQVRVR
#undef ELEM_SUB_VQVUVU
#undef ELEM_SUB_VQVIVI

!! subroutine s(__vector_quad, __vector_pair, vector(u))
#define ELEM_SUB_VQVPVU(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vpvu##VKIND(acc, a, b); \
    __vector_quad, intent(INTENT) :: acc; \
    __vector_pair, intent(in) :: a; \
    vector(unsigned(VKIND)), intent(in) :: b; \
  end subroutine ;

!! subroutine s(__vector_quad, __vector_pair, vector(r))
#define ELEM_SUB_VQVPVR(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vpvr##VKIND(acc, a, b); \
    __vector_quad, intent(INTENT) :: acc; \
    __vector_pair, intent(in) :: a; \
    vector(real(VKIND)), intent(in) :: b; \
  end subroutine ;

  ELEM_SUB_VQVPVU(inout,1)
  ELEM_SUB_VQVPVR(inout,8)
  ELEM_SUB_VQVPVU(out,1)
  ELEM_SUB_VQVPVR(out,8)

#undef ELEM_SUB_VQVPVR
#undef ELEM_SUB_VQVPVU

!! ========== 5 arguments subroutine interface ===============================!!
!! subroutine s(__vector_quad, vector(i), vector(i), vector(i), vector(i))
#define ELEM_SUB_VQVIVIVIVI(VKIND) \
  elemental subroutine sub_vqvi##VKIND##vi##VKIND##vi##VKIND##vi##VKIND(acc, arg1, arg2, arg3, arg4); \
    __vector_quad, intent(out) :: acc; \
    vector(integer(VKIND)), intent(in) :: arg1, arg2, arg3, arg4; \
  end subroutine ;

!! subroutine s(__vector_quad, vector(u), vector(u), vector(u), vector(u))
#define ELEM_SUB_VQVUVUVUVU(VKIND) \
  elemental subroutine sub_vqvu##VKIND##vu##VKIND##vu##VKIND##vu##VKIND(acc, arg1, arg2, arg3, arg4); \
    __vector_quad, intent(out) :: acc; \
    vector(unsigned(VKIND)), intent(in) :: arg1, arg2, arg3, arg4; \
  end subroutine ;

!! subroutine s(__vector_quad, vector(r), vector(r), vector(r), vector(r))
#define ELEM_SUB_VQVRVRVRVR(VKIND) \
  elemental subroutine sub_vqvr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND(acc, arg1, arg2, arg3, arg4); \
    __vector_quad, intent(out) :: acc; \
    vector(real(VKIND)), intent(in) :: arg1, arg2, arg3, arg4; \
  end subroutine ;

  ELEM_SUB_VQVIVIVIVI(1) ELEM_SUB_VQVIVIVIVI(2)
  ELEM_SUB_VQVIVIVIVI(4) ELEM_SUB_VQVIVIVIVI(8)
  ELEM_SUB_VQVUVUVUVU(1) ELEM_SUB_VQVUVUVUVU(2)
  ELEM_SUB_VQVUVUVUVU(4) ELEM_SUB_VQVUVUVUVU(8)
  ELEM_SUB_VQVRVRVRVR(4) ELEM_SUB_VQVRVRVRVR(8)

#undef ELEM_SUB_VQVRVRVRVR
#undef ELEM_SUB_VQVUVUVUVU
#undef ELEM_SUB_VQVIVIVIVI

!! subroutine s(__vector_quad, vector(u), vector(u), integer, integer)
#define ELEM_SUB_VQVUVUII(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vu##VKIND##vu##VKIND##ii(acc, a, b, xmask, ymask); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(unsigned(VKIND)), intent(in) :: a, b; \
    integer(8), intent(in) :: xmask, ymask; \
    !dir$ ignore_tkr(k) xmask; \
    !dir$ ignore_tkr(k) ymask; \
  end subroutine ;

!! subroutine s(__vector_quad, vector(r), vector(r), integer, integer)
#define ELEM_SUB_VQVRVRII(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vr##VKIND##vr##VKIND##ii(acc, a, b, xmask, ymask); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(real(VKIND)), intent(in) :: a, b; \
    integer(8), intent(in) :: xmask, ymask; \
    !dir$ ignore_tkr(k) xmask; \
    !dir$ ignore_tkr(k) ymask; \
  end subroutine ;

  ELEM_SUB_VQVUVUII(inout,1)
  ELEM_SUB_VQVRVRII(inout,4)
  ELEM_SUB_VQVUVUII(out,1)
  ELEM_SUB_VQVRVRII(out,4)

#undef ELEM_SUB_VQVRVRII
#undef ELEM_SUB_VQVUVUII

!! subroutine s(__vector_quad, __vector_pair, vector(u), integer, integer)
#define ELEM_SUB_VQVPVUII(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vpvu##VKIND##ii(acc, a, b, xmask, ymask); \
    __vector_quad, intent(INTENT) :: acc; \
    __vector_pair, intent(in) :: a; \
    vector(unsigned(VKIND)), intent(in) :: b; \
    integer(8), intent(in) :: xmask, ymask; \
    !dir$ ignore_tkr(k) xmask; \
    !dir$ ignore_tkr(k) ymask; \
  end subroutine ;

!! subroutine s(__vector_quad, __vector_pair, vector(r), integer, integer)
#define ELEM_SUB_VQVPVRII(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vpvr##VKIND##ii(acc, a, b, xmask, ymask); \
    __vector_quad, intent(INTENT) :: acc; \
    __vector_pair, intent(in) :: a; \
    vector(real(VKIND)), intent(in) :: b; \
    integer(8), intent(in) :: xmask, ymask; \
    !dir$ ignore_tkr(k) xmask; \
    !dir$ ignore_tkr(k) ymask; \
  end subroutine ;

  ELEM_SUB_VQVPVUII(inout,1)
  ELEM_SUB_VQVPVRII(inout,8)
  ELEM_SUB_VQVPVUII(out,1)
  ELEM_SUB_VQVPVRII(out,8)

#undef ELEM_SUB_VQVPVRII
#undef ELEM_SUB_VQVPVUII

!! ========== 6 arguments subroutine interface ===============================!!
!! subroutine s(__vector_quad, vector(i), vector(i), integer, integer, integer)
#define ELEM_SUB_VQVIVIIII(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vi##VKIND##vi##VKIND##iii(acc, a, b, xmask, ymask, pmask); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(integer(VKIND)), intent(in) :: a, b; \
    integer(8), intent(in) :: xmask, ymask, pmask; \
    !dir$ ignore_tkr(k) xmask; \
    !dir$ ignore_tkr(k) ymask; \
    !dir$ ignore_tkr(k) pmask; \
  end subroutine ;

!! subroutine s(__vector_quad, vector(u), vector(u), integer, integer, integer)
#define ELEM_SUB_VQVUVUIII(INTENT, VKIND) \
  elemental subroutine sub_vq##INTENT##vu##VKIND##vu##VKIND##iii(acc, a, b, xmask, ymask, pmask); \
    __vector_quad, intent(INTENT) :: acc; \
    vector(unsigned(VKIND)), intent(in) :: a, b; \
    integer(8), intent(in) :: xmask, ymask, pmask; \
    !dir$ ignore_tkr(k) xmask; \
    !dir$ ignore_tkr(k) ymask; \
    !dir$ ignore_tkr(k) pmask; \
  end subroutine ;

  ELEM_SUB_VQVIVIIII(inout,1) ELEM_SUB_VQVIVIIII(inout,2)
  ELEM_SUB_VQVUVUIII(inout,1)
  ELEM_SUB_VQVIVIIII(out,1) ELEM_SUB_VQVIVIIII(out,2)
  ELEM_SUB_VQVUVUIII(out,1)

#undef ELEM_SUB_VQVUVUIII
#undef ELEM_SUB_VQVIVIIII

!! ========== non-macro interface =============================================!!
  elemental subroutine sub_atvp(data, pair)
    ! Dummy arg 'data' is supposed to be intent(out) of any type,
    ! but according to Fortran 2018: C709: Type(*) arguments can not have
    ! intent(out) attribute. Use intent(inout) instead.
    type(*), intent(inout) :: data
    __vector_pair, intent(inout) :: pair
  end subroutine

  elemental subroutine sub_atvq(data, acc)
    ! Dummy arg 'data' is supposed to be intent(out) of any type,
    ! but according to Fortran 2018: C709: Type(*) arguments can not have
    ! intent(out) attribute. Use intent(inout) instead.
    type(*), intent(inout) :: data
    __vector_quad, intent(inout) :: acc
  end subroutine

  end interface

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

! mma_lxvp (using vec_lxvp)
  VEC_VP_I0_VI(vec_lxvp,1) VEC_VP_I0_VI(vec_lxvp,2) VEC_VP_I0_VI(vec_lxvp,4) VEC_VP_I0_VI(vec_lxvp,8)
  VEC_VP_I0_VU(vec_lxvp,1) VEC_VP_I0_VU(vec_lxvp,2) VEC_VP_I0_VU(vec_lxvp,4) VEC_VP_I0_VU(vec_lxvp,8)
  VEC_VP_I0_VR(vec_lxvp,4) VEC_VP_I0_VR(vec_lxvp,8)
  VEC_VP_I0_VP(vec_lxvp)
  interface mma_lxvp
     procedure :: VP_I0_VI(vec_lxvp,1), VP_I0_VI(vec_lxvp,2), VP_I0_VI(vec_lxvp,4), VP_I0_VI(vec_lxvp,8)
     procedure :: VP_I0_VU(vec_lxvp,1), VP_I0_VU(vec_lxvp,2), VP_I0_VU(vec_lxvp,4), VP_I0_VU(vec_lxvp,8)
     procedure :: VP_I0_VR(vec_lxvp,4), VP_I0_VR(vec_lxvp,8)
     procedure :: VP_I0_VP(vec_lxvp)
  end interface mma_lxvp
  public :: mma_lxvp

#undef VEC_VP_I0_VP
#undef VEC_VP_I0_VR
#undef VEC_VP_I0_VU
#undef VEC_VP_I0_VI
#undef VP_I0_VP
#undef VP_I0_VR
#undef VP_I0_VU
#undef VP_I0_VI

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

! mma_stxvp (using vec_stxvp)
  procedure(sub_vpi0vp) :: __ppc_vec_stxvp_vpi0vp0
  procedure(sub_vpi0i0) :: __ppc_vec_stxvp_vpi0i0
  procedure(sub_vpi0r0) :: __ppc_vec_stxvp_vpi0r0
  VEC_VP_I0_VI(vec_stxvp, 1) VEC_VP_I0_VI(vec_stxvp, 2) VEC_VP_I0_VI(vec_stxvp, 4) VEC_VP_I0_VI(vec_stxvp, 8)
  VEC_VP_I0_VU(vec_stxvp, 1) VEC_VP_I0_VU(vec_stxvp, 2) VEC_VP_I0_VU(vec_stxvp, 4) VEC_VP_I0_VU(vec_stxvp, 8)
  VEC_VP_I0_VR(vec_stxvp, 4) VEC_VP_I0_VR(vec_stxvp, 8)
  interface mma_stxvp
     procedure :: __ppc_vec_stxvp_vpi0vp0
     procedure :: __ppc_vec_stxvp_vpi0i0
     procedure :: __ppc_vec_stxvp_vpi0r0
     procedure :: VP_I0_VI(vec_stxvp, 1), VP_I0_VI(vec_stxvp, 2), VP_I0_VI(vec_stxvp, 4), VP_I0_VI(vec_stxvp, 8)
     procedure :: VP_I0_VU(vec_stxvp, 1), VP_I0_VU(vec_stxvp, 2), VP_I0_VU(vec_stxvp, 4), VP_I0_VU(vec_stxvp, 8)
     procedure :: VP_I0_VR(vec_stxvp, 4), VP_I0_VR(vec_stxvp, 8)
  end interface mma_stxvp
  public :: mma_stxvp

#undef VEC_VP_I0_VR
#undef VEC_VP_I0_VU
#undef VEC_VP_I0_VI
#undef VP_I0_VR
#undef VP_I0_VU
#undef VP_I0_VI

#define SUB_VQ_VI_VI_VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND##vi##VKIND##vi##VKIND
#define SUB_VQ_VU_VU_VU_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND##vu##VKIND##vu##VKIND
#define SUB_VQ_VR_VR_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND

#define VEC_SUB_VQ_VI_VI_VI_VI(NAME, VKIND) \
  procedure(sub_vqvi##VKIND##vi##VKIND##vi##VKIND##vi##VKIND) :: SUB_VQ_VI_VI_VI_VI(NAME, VKIND);
#define VEC_SUB_VQ_VU_VU_VU_VU(NAME, VKIND) \
  procedure(sub_vqvu##VKIND##vu##VKIND##vu##VKIND##vu##VKIND) :: SUB_VQ_VU_VU_VU_VU(NAME, VKIND);
#define VEC_SUB_VQ_VR_VR_VR_VR(NAME, VKIND) \
  procedure(sub_vqvr##VKIND##vr##VKIND##vr##VKIND##vr##VKIND) :: SUB_VQ_VR_VR_VR_VR(NAME, VKIND);

! mma_assemble_acc
  VEC_SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,1)
  VEC_SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,2)
  VEC_SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,4)
  VEC_SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,8)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,1)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,2)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,4)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,8)
  VEC_SUB_VQ_VR_VR_VR_VR(mma_assemble_acc,4)
  VEC_SUB_VQ_VR_VR_VR_VR(mma_assemble_acc,8)
  interface mma_assemble_acc
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,1)
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,2)
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,4)
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_assemble_acc,8)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,1)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,2)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,4)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_assemble_acc,8)
    procedure :: SUB_VQ_VR_VR_VR_VR(mma_assemble_acc,4)
    procedure :: SUB_VQ_VR_VR_VR_VR(mma_assemble_acc,8)
  end interface
  public mma_assemble_acc

! mma_build_acc
  VEC_SUB_VQ_VI_VI_VI_VI(mma_build_acc,1)
  VEC_SUB_VQ_VI_VI_VI_VI(mma_build_acc,2)
  VEC_SUB_VQ_VI_VI_VI_VI(mma_build_acc,4)
  VEC_SUB_VQ_VI_VI_VI_VI(mma_build_acc,8)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_build_acc,1)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_build_acc,2)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_build_acc,4)
  VEC_SUB_VQ_VU_VU_VU_VU(mma_build_acc,8)
  VEC_SUB_VQ_VR_VR_VR_VR(mma_build_acc,4)
  VEC_SUB_VQ_VR_VR_VR_VR(mma_build_acc,8)
  interface mma_build_acc
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_build_acc,1)
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_build_acc,2)
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_build_acc,4)
    procedure :: SUB_VQ_VI_VI_VI_VI(mma_build_acc,8)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_build_acc,1)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_build_acc,2)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_build_acc,4)
    procedure :: SUB_VQ_VU_VU_VU_VU(mma_build_acc,8)
    procedure :: SUB_VQ_VR_VR_VR_VR(mma_build_acc,4)
    procedure :: SUB_VQ_VR_VR_VR_VR(mma_build_acc,8)
  end interface
  public mma_build_acc

#undef VEC_SUB_VQ_VR_VR_VR_VR
#undef VEC_SUB_VQ_VU_VU_VU_VU
#undef VEC_SUB_VQ_VI_VI_VI_VI
#undef SUB_VQ_VR_VR_VR_VR
#undef SUB_VQ_VU_VU_VU_VU
#undef SUB_VQ_VI_VI_VI_VI

#define SUB_VP_VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND
#define SUB_VP_VU_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND
#define SUB_VP_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND

#define VEC_SUB_VP_VI_VI(NAME, VKIND) \
  procedure(sub_vpvi##VKIND##vi##VKIND) :: SUB_VP_VI_VI(NAME, VKIND);
#define VEC_SUB_VP_VU_VU(NAME, VKIND) \
  procedure(sub_vpvu##VKIND##vu##VKIND) :: SUB_VP_VU_VU(NAME, VKIND);
#define VEC_SUB_VP_VR_VR(NAME, VKIND) \
  procedure(sub_vpvr##VKIND##vr##VKIND) :: SUB_VP_VR_VR(NAME, VKIND);

! mma_assemble_pair
  VEC_SUB_VP_VI_VI(mma_assemble_pair,1) VEC_SUB_VP_VI_VI(mma_assemble_pair,2)
  VEC_SUB_VP_VI_VI(mma_assemble_pair,4) VEC_SUB_VP_VI_VI(mma_assemble_pair,8)
  VEC_SUB_VP_VU_VU(mma_assemble_pair,1) VEC_SUB_VP_VU_VU(mma_assemble_pair,2)
  VEC_SUB_VP_VU_VU(mma_assemble_pair,4) VEC_SUB_VP_VU_VU(mma_assemble_pair,8)
  VEC_SUB_VP_VR_VR(mma_assemble_pair,4) VEC_SUB_VP_VR_VR(mma_assemble_pair,8)
  interface mma_assemble_pair
    procedure :: SUB_VP_VI_VI(mma_assemble_pair,1)
    procedure :: SUB_VP_VI_VI(mma_assemble_pair,2)
    procedure :: SUB_VP_VI_VI(mma_assemble_pair,4)
    procedure :: SUB_VP_VI_VI(mma_assemble_pair,8)
    procedure :: SUB_VP_VU_VU(mma_assemble_pair,1)
    procedure :: SUB_VP_VU_VU(mma_assemble_pair,2)
    procedure :: SUB_VP_VU_VU(mma_assemble_pair,4)
    procedure :: SUB_VP_VU_VU(mma_assemble_pair,8)
    procedure :: SUB_VP_VR_VR(mma_assemble_pair,4)
    procedure :: SUB_VP_VR_VR(mma_assemble_pair,8)
  end interface
  public mma_assemble_pair

#undef VEC_SUB_VP_VR_VR
#undef VEC_SUB_VP_VU_VU
#undef VEC_SUB_VP_VI_VI
#undef SUB_VP_VR_VR
#undef SUB_VP_VU_VU
#undef SUB_VP_VI_VI

#define SUB_VQ_VI_VI_I_I_I(NAME, VKIND) __ppc_##NAME##_vqvi##VKIND##vi##VKINDi0i0i0
#define SUB_VQ_VU_VU_I_I_I(NAME, VKIND) __ppc_##NAME##_vqvu##VKIND##vu##VKINDi0i0ii0

#define VEC_SUB_VQ_VI_VI_I_I_I(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vi##VKIND##vi##VKIND##iii) :: SUB_VQ_VI_VI_I_I_I(NAME, VKIND);
#define VEC_SUB_VQ_VU_VU_I_I_I(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vu##VKIND##vu##VKIND##iii) :: SUB_VQ_VU_VU_I_I_I(NAME, VKIND);

! mma_pmxvbf16ger2
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2_,out,1)
  interface mma_pmxvbf16ger2
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2_,1)
  end interface mma_pmxvbf16ger2
  public mma_pmxvbf16ger2

! mma_pmxvbf16ger2nn
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2nn,inout,1)
  interface mma_pmxvbf16ger2nn
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2nn,1)
  end interface mma_pmxvbf16ger2nn
  public mma_pmxvbf16ger2nn

! mma_pmxvbf16ger2np
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2np,inout,1)
  interface mma_pmxvbf16ger2np
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2np,1)
  end interface mma_pmxvbf16ger2np
  public mma_pmxvbf16ger2np

! mma_pmxvbf16ger2pn
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2pn,inout,1)
  interface mma_pmxvbf16ger2pn
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2pn,1)
  end interface mma_pmxvbf16ger2pn
  public mma_pmxvbf16ger2pn

! mma_pmxvbf16ger2pp
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2pp,inout,1)
  interface mma_pmxvbf16ger2pp
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvbf16ger2pp,1)
  end interface mma_pmxvbf16ger2pp
  public mma_pmxvbf16ger2pp

! mma_pmxvf16ger2
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2_,out,1)
  interface mma_pmxvf16ger2
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2_,1)
  end interface mma_pmxvf16ger2
  public mma_pmxvf16ger2

! mma_pmxvf16ger2nn
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2nn,inout,1)
  interface mma_pmxvf16ger2nn
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2nn,1)
  end interface mma_pmxvf16ger2nn
  public mma_pmxvf16ger2nn

! mma_pmxvf16ger2np
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2np,inout,1)
  interface mma_pmxvf16ger2np
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2np,1)
  end interface mma_pmxvf16ger2np
  public mma_pmxvf16ger2np

! mma_pmxvf16ger2pn
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2pn,inout,1)
  interface mma_pmxvf16ger2pn
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2pn,1)
  end interface mma_pmxvf16ger2pn
  public mma_pmxvf16ger2pn

! mma_pmxvf16ger2pp
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2pp,inout,1)
  interface mma_pmxvf16ger2pp
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvf16ger2pp,1)
  end interface mma_pmxvf16ger2pp
  public mma_pmxvf16ger2pp

! mma_pmxvi16ger2
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2_,out,2)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2_,out,1)
  interface mma_pmxvi16ger2
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2_,2)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2_,1)
  end interface mma_pmxvi16ger2
  public mma_pmxvi16ger2

! mma_pmxvi16ger2pp
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2pp,inout,2)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2pp,inout,1)
  interface mma_pmxvi16ger2pp
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2pp,2)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2pp,1)
  end interface mma_pmxvi16ger2pp
  public mma_pmxvi16ger2pp

! mma_pmxvi16ger2s
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2s,out,2)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2s,out,1)
  interface mma_pmxvi16ger2s
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2s,2)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2s,1)
  end interface mma_pmxvi16ger2s
  public mma_pmxvi16ger2s

! mma_pmxvi16ger2spp
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2spp,inout,2)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2spp,inout,1)
  interface mma_pmxvi16ger2spp
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi16ger2spp,2)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi16ger2spp,1)
  end interface mma_pmxvi16ger2spp
  public mma_pmxvi16ger2spp

! mma_pmxvi4ger8
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi4ger8_,out,1)
  interface mma_pmxvi4ger8
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi4ger8_,1)
  end interface mma_pmxvi4ger8
  public mma_pmxvi4ger8

! mma_pmxvi4ger8pp
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi4ger8pp,inout,1)
  interface mma_pmxvi4ger8pp
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi4ger8pp,1)
  end interface mma_pmxvi4ger8pp
  public mma_pmxvi4ger8pp

! mma_pmxvi8ger4
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi8ger4_,out,1)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi8ger4_,out,1)
  interface mma_pmxvi8ger4
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi8ger4_,1)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi8ger4_,1)
  end interface mma_pmxvi8ger4
  public mma_pmxvi8ger4

! mma_pmxvi8ger4pp
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi8ger4pp,inout,1)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi8ger4pp,inout,1)
  interface mma_pmxvi8ger4pp
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi8ger4pp,1)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi8ger4pp,1)
  end interface mma_pmxvi8ger4pp
  public mma_pmxvi8ger4pp

! mma_pmxvi8ger4spp
  VEC_SUB_VQ_VI_VI_I_I_I(mma_pmxvi8ger4spp,inout,1)
  VEC_SUB_VQ_VU_VU_I_I_I(mma_pmxvi8ger4spp,inout,1)
  interface mma_pmxvi8ger4spp
    procedure :: SUB_VQ_VI_VI_I_I_I(mma_pmxvi8ger4spp,1)
    procedure :: SUB_VQ_VU_VU_I_I_I(mma_pmxvi8ger4spp,1)
  end interface mma_pmxvi8ger4spp
  public mma_pmxvi8ger4spp

#undef VEC_SUB_VQ_VU_VU_I_I_I
#undef VEC_SUB_VQ_VI_VI_I_I_I
#undef SUB_VQ_VU_VU_I_I_I
#undef SUB_VQ_VI_VI_I_I_I

#define SUB_VQ_VU_VU_I_I(NAME, VKIND) __ppc_##NAME##_vqvu##VKIND##vu##VKINDi0i0
#define SUB_VQ_VR_VR_I_I(NAME, VKIND) __ppc_##NAME##_vqvr##VKIND##vr##VKINDi0i0

#define VEC_SUB_VQ_VU_VU_I_I(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vu##VKIND##vu##VKIND##ii) :: SUB_VQ_VU_VU_I_I(NAME, VKIND);
#define VEC_SUB_VQ_VR_VR_I_I(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vr##VKIND##vr##VKIND##ii) :: SUB_VQ_VR_VR_I_I(NAME, VKIND);

! mma_pmxvf32ger
  VEC_SUB_VQ_VU_VU_I_I(mma_pmxvf32ger,out,1)
  VEC_SUB_VQ_VR_VR_I_I(mma_pmxvf32ger,out,4)
  interface mma_pmxvf32ger
    procedure :: SUB_VQ_VU_VU_I_I(mma_pmxvf32ger,1)
    procedure :: SUB_VQ_VR_VR_I_I(mma_pmxvf32ger,4)
  end interface mma_pmxvf32ger
  public mma_pmxvf32ger

! mma_pmxvf32gernn
  VEC_SUB_VQ_VU_VU_I_I(mma_pmxvf32gernn,inout,1)
  VEC_SUB_VQ_VR_VR_I_I(mma_pmxvf32gernn,inout,4)
  interface mma_pmxvf32gernn
    procedure :: SUB_VQ_VU_VU_I_I(mma_pmxvf32gernn,1)
    procedure :: SUB_VQ_VR_VR_I_I(mma_pmxvf32gernn,4)
  end interface mma_pmxvf32gernn
  public mma_pmxvf32gernn

! mma_pmxvf32gernp
  VEC_SUB_VQ_VU_VU_I_I(mma_pmxvf32gernp,inout,1)
  VEC_SUB_VQ_VR_VR_I_I(mma_pmxvf32gernp,inout,4)
  interface mma_pmxvf32gernp
    procedure :: SUB_VQ_VU_VU_I_I(mma_pmxvf32gernp,1)
    procedure :: SUB_VQ_VR_VR_I_I(mma_pmxvf32gernp,4)
  end interface mma_pmxvf32gernp
  public mma_pmxvf32gernp

! mma_pmxvf32gerpn
  VEC_SUB_VQ_VU_VU_I_I(mma_pmxvf32gerpn,inout,1)
  VEC_SUB_VQ_VR_VR_I_I(mma_pmxvf32gerpn,inout,4)
  interface mma_pmxvf32gerpn
    procedure :: SUB_VQ_VU_VU_I_I(mma_pmxvf32gerpn,1)
    procedure :: SUB_VQ_VR_VR_I_I(mma_pmxvf32gerpn,4)
  end interface mma_pmxvf32gerpn
  public mma_pmxvf32gerpn

! mma_pmxvf32gerpp
  VEC_SUB_VQ_VU_VU_I_I(mma_pmxvf32gerpp,inout,1)
  VEC_SUB_VQ_VR_VR_I_I(mma_pmxvf32gerpp,inout,4)
  interface mma_pmxvf32gerpp
    procedure :: SUB_VQ_VU_VU_I_I(mma_pmxvf32gerpp,1)
    procedure :: SUB_VQ_VR_VR_I_I(mma_pmxvf32gerpp,4)
  end interface mma_pmxvf32gerpp
  public mma_pmxvf32gerpp

#undef VEC_SUB_VQ_VR_VR_I_I
#undef VEC_SUB_VQ_VU_VU_I_I
#undef SUB_VQ_VR_VR_I_I
#undef SUB_VQ_VU_VU_I_I

#define SUB_VQ_VP_VU_I_I(NAME, VKIND) __ppc_##NAME##_vqvpvu##VKINDi0i0
#define SUB_VQ_VP_VR_I_I(NAME, VKIND) __ppc_##NAME##_vqvpvr##VKINDi0i0

#define VEC_SUB_VQ_VP_VU_I_I(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vpvu##VKIND##ii) :: SUB_VQ_VP_VU_I_I(NAME, VKIND);
#define VEC_SUB_VQ_VP_VR_I_I(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vpvr##VKIND##ii) :: SUB_VQ_VP_VR_I_I(NAME, VKIND);

! mma_pmxvf64ger
  VEC_SUB_VQ_VP_VU_I_I(mma_pmxvf64ger,out,1)
  VEC_SUB_VQ_VP_VR_I_I(mma_pmxvf64ger,out,8)
  interface mma_pmxvf64ger
    procedure :: SUB_VQ_VP_VU_I_I(mma_pmxvf64ger,1)
    procedure :: SUB_VQ_VP_VR_I_I(mma_pmxvf64ger,8)
  end interface mma_pmxvf64ger
  public mma_pmxvf64ger

! mma_pmxvf64gernn
  VEC_SUB_VQ_VP_VU_I_I(mma_pmxvf64gernn,inout,1)
  VEC_SUB_VQ_VP_VR_I_I(mma_pmxvf64gernn,inout,8)
  interface mma_pmxvf64gernn
    procedure :: SUB_VQ_VP_VU_I_I(mma_pmxvf64gernn,1)
    procedure :: SUB_VQ_VP_VR_I_I(mma_pmxvf64gernn,8)
  end interface mma_pmxvf64gernn
  public mma_pmxvf64gernn

! mma_pmxvf64gernp
  VEC_SUB_VQ_VP_VU_I_I(mma_pmxvf64gernp,inout,1)
  VEC_SUB_VQ_VP_VR_I_I(mma_pmxvf64gernp,inout,8)
  interface mma_pmxvf64gernp
    procedure :: SUB_VQ_VP_VU_I_I(mma_pmxvf64gernp,1)
    procedure :: SUB_VQ_VP_VR_I_I(mma_pmxvf64gernp,8)
  end interface mma_pmxvf64gernp
  public mma_pmxvf64gernp

! mma_pmxvf64gerpn
  VEC_SUB_VQ_VP_VU_I_I(mma_pmxvf64gerpn,inout,1)
  VEC_SUB_VQ_VP_VR_I_I(mma_pmxvf64gerpn,inout,8)
  interface mma_pmxvf64gerpn
    procedure :: SUB_VQ_VP_VU_I_I(mma_pmxvf64gerpn,1)
    procedure :: SUB_VQ_VP_VR_I_I(mma_pmxvf64gerpn,8)
  end interface mma_pmxvf64gerpn
  public mma_pmxvf64gerpn

! mma_pmxvf64gerpp
  VEC_SUB_VQ_VP_VU_I_I(mma_pmxvf64gerpp,inout,1)
  VEC_SUB_VQ_VP_VR_I_I(mma_pmxvf64gerpp,inout,8)
  interface mma_pmxvf64gerpp
    procedure :: SUB_VQ_VP_VU_I_I(mma_pmxvf64gerpp,1)
    procedure :: SUB_VQ_VP_VR_I_I(mma_pmxvf64gerpp,8)
  end interface mma_pmxvf64gerpp
  public mma_pmxvf64gerpp

#undef VEC_SUB_VQ_VP_VR_I_I
#undef VEC_SUB_VQ_VP_VU_I_I
#undef SUB_VQ_VP_VR_I_I
#undef SUB_VQ_VP_VU_I_I

#define SUB_VQ_VI_VI(NAME, VKIND) __ppc_##NAME##_vi##VKIND##vi##VKIND
#define SUB_VQ_VU_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND##vu##VKIND
#define SUB_VQ_VR_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND##vr##VKIND

#define VEC_SUB_VQ_VI_VI(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vi##VKIND##vi##VKIND) :: SUB_VQ_VI_VI(NAME, VKIND);
#define VEC_SUB_VQ_VU_VU(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vu##VKIND##vu##VKIND) :: SUB_VQ_VU_VU(NAME, VKIND);
#define VEC_SUB_VQ_VR_VR(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vr##VKIND##vr##VKIND) :: SUB_VQ_VR_VR(NAME, VKIND);

!! First argument with INTENT(INOUT)
! mma_xvbf16ger2nn
  VEC_SUB_VQ_VU_VU(mma_xvbf16ger2nn,inout,1)
  interface mma_xvbf16ger2nn
    procedure :: SUB_VQ_VU_VU(mma_xvbf16ger2nn,1)
  end interface
  public mma_xvbf16ger2nn

! mma_xvbf16ger2np
  VEC_SUB_VQ_VU_VU(mma_xvbf16ger2np,inout,1)
  interface mma_xvbf16ger2np
    procedure :: SUB_VQ_VU_VU(mma_xvbf16ger2np,1)
  end interface
  public mma_xvbf16ger2np

! mma_xvbf16ger2pn
  VEC_SUB_VQ_VU_VU(mma_xvbf16ger2pn,inout,1)
  interface mma_xvbf16ger2pn
    procedure :: SUB_VQ_VU_VU(mma_xvbf16ger2pn,1)
  end interface
  public mma_xvbf16ger2pn

! mma_xvbf16ger2pp
  VEC_SUB_VQ_VU_VU(mma_xvbf16ger2pp,inout,1)
  interface mma_xvbf16ger2pp
    procedure :: SUB_VQ_VU_VU(mma_xvbf16ger2pp,1)
  end interface
  public mma_xvbf16ger2pp

! mma_xvi8ger4pp
  VEC_SUB_VQ_VI_VI(mma_xvi8ger4pp,inout,1)
  VEC_SUB_VQ_VU_VU(mma_xvi8ger4pp,inout,1)
  interface mma_xvi8ger4pp
    procedure :: SUB_VQ_VI_VI(mma_xvi8ger4pp,1)
    procedure :: SUB_VQ_VU_VU(mma_xvi8ger4pp,1)
  end interface
  public mma_xvi8ger4pp

! mma_xvi8ger4spp
  VEC_SUB_VQ_VI_VI(mma_xvi8ger4spp,inout,1)
  VEC_SUB_VQ_VU_VU(mma_xvi8ger4spp,inout,1)
  interface mma_xvi8ger4spp
    procedure :: SUB_VQ_VI_VI(mma_xvi8ger4spp,1)
    procedure :: SUB_VQ_VU_VU(mma_xvi8ger4spp,1)
  end interface
  public mma_xvi8ger4spp

! mma_xvi16ger2pp
  VEC_SUB_VQ_VI_VI(mma_xvi16ger2pp,inout,2)
  VEC_SUB_VQ_VU_VU(mma_xvi16ger2pp,inout,1)
  interface mma_xvi16ger2pp
    procedure :: SUB_VQ_VI_VI(mma_xvi16ger2pp,2)
    procedure :: SUB_VQ_VU_VU(mma_xvi16ger2pp,1)
  end interface
  public mma_xvi16ger2pp

! mma_xvi16ger2s
  VEC_SUB_VQ_VI_VI(mma_xvi16ger2s,inout,2)
  VEC_SUB_VQ_VU_VU(mma_xvi16ger2s,inout,1)
  interface mma_xvi16ger2s
    procedure :: SUB_VQ_VI_VI(mma_xvi16ger2s,2)
    procedure :: SUB_VQ_VU_VU(mma_xvi16ger2s,1)
  end interface
  public mma_xvi16ger2s

! mma_xvi16ger2spp
  VEC_SUB_VQ_VI_VI(mma_xvi16ger2spp,inout,2)
  VEC_SUB_VQ_VU_VU(mma_xvi16ger2spp,inout,1)
  interface mma_xvi16ger2spp
    procedure :: SUB_VQ_VI_VI(mma_xvi16ger2spp,2)
    procedure :: SUB_VQ_VU_VU(mma_xvi16ger2spp,1)
  end interface
  public mma_xvi16ger2spp

! mma_xvi4ger8pp
  VEC_SUB_VQ_VU_VU(mma_xvi4ger8pp,inout,1)
  interface mma_xvi4ger8pp
    procedure :: SUB_VQ_VU_VU(mma_xvi4ger8pp,1)
  end interface
  public mma_xvi4ger8pp

! mma_xvf16ger2nn
  VEC_SUB_VQ_VU_VU(mma_xvf16ger2nn,inout,1)
  interface mma_xvf16ger2nn
    procedure :: SUB_VQ_VU_VU(mma_xvf16ger2nn,1)
  end interface
  public mma_xvf16ger2nn

! mma_xvf16ger2np
  VEC_SUB_VQ_VU_VU(mma_xvf16ger2np,inout,1)
  interface mma_xvf16ger2np
    procedure :: SUB_VQ_VU_VU(mma_xvf16ger2np,1)
  end interface
  public mma_xvf16ger2np

! mma_xvf16ger2pn
  VEC_SUB_VQ_VU_VU(mma_xvf16ger2pn,inout,1)
  interface mma_xvf16ger2pn
    procedure :: SUB_VQ_VU_VU(mma_xvf16ger2pn,1)
  end interface
  public mma_xvf16ger2pn

! mma_xvf16ger2pp
  VEC_SUB_VQ_VU_VU(mma_xvf16ger2pp,inout,1)
  interface mma_xvf16ger2pp
    procedure :: SUB_VQ_VU_VU(mma_xvf16ger2pp,1)
  end interface
  public mma_xvf16ger2pp

! mma_xvf32gernn
  VEC_SUB_VQ_VU_VU(mma_xvf32gernn,inout,1)
  VEC_SUB_VQ_VR_VR(mma_xvf32gernn,inout,4)
  interface mma_xvf32gernn
    procedure :: SUB_VQ_VU_VU(mma_xvf32gernn,1)
    procedure :: SUB_VQ_VR_VR(mma_xvf32gernn,4)
  end interface
  public mma_xvf32gernn

! mma_xvf32gernp
  VEC_SUB_VQ_VU_VU(mma_xvf32gernp,inout,1)
  VEC_SUB_VQ_VR_VR(mma_xvf32gernp,inout,4)
  interface mma_xvf32gernp
    procedure :: SUB_VQ_VU_VU(mma_xvf32gernp,1)
    procedure :: SUB_VQ_VR_VR(mma_xvf32gernp,4)
  end interface
  public mma_xvf32gernp

! mma_xvf32gerpn
  VEC_SUB_VQ_VU_VU(mma_xvf32gerpn,inout,1)
  VEC_SUB_VQ_VR_VR(mma_xvf32gerpn,inout,4)
  interface mma_xvf32gerpn
    procedure :: SUB_VQ_VU_VU(mma_xvf32gerpn,1)
    procedure :: SUB_VQ_VR_VR(mma_xvf32gerpn,4)
  end interface
  public mma_xvf32gerpn

! mma_xvf32gerpp
  VEC_SUB_VQ_VU_VU(mma_xvf32gerpp,inout,1)
  VEC_SUB_VQ_VR_VR(mma_xvf32gerpp,inout,4)
  interface mma_xvf32gerpp
    procedure :: SUB_VQ_VU_VU(mma_xvf32gerpp,1)
    procedure :: SUB_VQ_VR_VR(mma_xvf32gerpp,4)
  end interface
  public mma_xvf32gerpp

!! First argument with INTENT(OUT)
! mma_xvbf16ger2
  VEC_SUB_VQ_VU_VU(mma_xvbf16ger2_,out,1)
  interface mma_xvbf16ger2
    procedure :: SUB_VQ_VU_VU(mma_xvbf16ger2_,1)
  end interface
  public mma_xvbf16ger2

! mma_xvi16ger2
  VEC_SUB_VQ_VI_VI(mma_xvi16ger2_,out,2)
  VEC_SUB_VQ_VU_VU(mma_xvi16ger2_,out,1)
  interface mma_xvi16ger2
    procedure :: SUB_VQ_VI_VI(mma_xvi16ger2_,2)
    procedure :: SUB_VQ_VU_VU(mma_xvi16ger2_,1)
  end interface
  public mma_xvi16ger2

! mma_xvi4ger8
  VEC_SUB_VQ_VU_VU(mma_xvi4ger8_,out,1)
  interface mma_xvi4ger8
    procedure :: SUB_VQ_VU_VU(mma_xvi4ger8_,1)
  end interface
  public mma_xvi4ger8

! mma_xvi8ger4
  VEC_SUB_VQ_VI_VI(mma_xvi8ger4_,out,1)
  VEC_SUB_VQ_VU_VU(mma_xvi8ger4_,out,1)
  interface mma_xvi8ger4
    procedure :: SUB_VQ_VI_VI(mma_xvi8ger4_,1)
    procedure :: SUB_VQ_VU_VU(mma_xvi8ger4_,1)
  end interface
  public mma_xvi8ger4

! mma_xvf16ger2
  VEC_SUB_VQ_VU_VU(mma_xvf16ger2_,out,1)
  interface mma_xvf16ger2
    procedure :: SUB_VQ_VU_VU(mma_xvf16ger2_,1)
  end interface
  public mma_xvf16ger2

! mma_xvf32ger
  VEC_SUB_VQ_VU_VU(mma_xvf32ger,out,1)
  VEC_SUB_VQ_VR_VR(mma_xvf32ger,out,4)
  interface mma_xvf32ger
    procedure :: SUB_VQ_VU_VU(mma_xvf32ger,1)
    procedure :: SUB_VQ_VR_VR(mma_xvf32ger,4)
  end interface
  public mma_xvf32ger

#undef VEC_SUB_VQ_VR_VR
#undef VEC_SUB_VQ_VU_VU
#undef VEC_SUB_VQ_VI_VI
#undef SUB_VQ_VR_VR
#undef SUB_VQ_VU_VU
#undef SUB_VQ_VI_VI

#define SUB_VQ_VP_VU(NAME, VKIND) __ppc_##NAME##_vu##VKIND
#define SUB_VQ_VP_VR(NAME, VKIND) __ppc_##NAME##_vr##VKIND

#define VEC_SUB_VQ_VP_VU(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vpvu##VKIND) :: SUB_VQ_VP_VU(NAME, VKIND);
#define VEC_SUB_VQ_VP_VR(NAME, INTENT, VKIND) \
  procedure(sub_vq##INTENT##vpvr##VKIND) :: SUB_VQ_VP_VR(NAME, VKIND);

! mma_xvf64ger
  VEC_SUB_VQ_VP_VU(mma_xvf64ger,out,1)
  VEC_SUB_VQ_VP_VR(mma_xvf64ger,out,8)
  interface mma_xvf64ger
    procedure :: SUB_VQ_VP_VU(mma_xvf64ger,1)
    procedure :: SUB_VQ_VP_VR(mma_xvf64ger,8)
  end interface
  public mma_xvf64ger

! mma_xvf64gernn
  VEC_SUB_VQ_VP_VU(mma_xvf64gernn,inout,1)
  VEC_SUB_VQ_VP_VR(mma_xvf64gernn,inout,8)
  interface mma_xvf64gernn
    procedure :: SUB_VQ_VP_VU(mma_xvf64gernn,1)
    procedure :: SUB_VQ_VP_VR(mma_xvf64gernn,8)
  end interface
  public mma_xvf64gernn

! mma_xvf64gernp
  VEC_SUB_VQ_VP_VU(mma_xvf64gernp,inout,1)
  VEC_SUB_VQ_VP_VR(mma_xvf64gernp,inout,8)
  interface mma_xvf64gernp
    procedure :: SUB_VQ_VP_VU(mma_xvf64gernp,1)
    procedure :: SUB_VQ_VP_VR(mma_xvf64gernp,8)
  end interface
  public mma_xvf64gernp

! mma_xvf64gerpn
  VEC_SUB_VQ_VP_VU(mma_xvf64gerpn,inout,1)
  VEC_SUB_VQ_VP_VR(mma_xvf64gerpn,inout,8)
  interface mma_xvf64gerpn
    procedure :: SUB_VQ_VP_VU(mma_xvf64gerpn,1)
    procedure :: SUB_VQ_VP_VR(mma_xvf64gerpn,8)
  end interface
  public mma_xvf64gerpn

! mma_xvf64gerpp
  VEC_SUB_VQ_VP_VU(mma_xvf64gerpp,inout,1)
  VEC_SUB_VQ_VP_VR(mma_xvf64gerpp,inout,8)
  interface mma_xvf64gerpp
    procedure :: SUB_VQ_VP_VU(mma_xvf64gerpp,1)
    procedure :: SUB_VQ_VP_VR(mma_xvf64gerpp,8)
  end interface
  public mma_xvf64gerpp

#undef VEC_SUB_VQ_VP_VR
#undef VEC_SUB_VQ_VP_VU
#undef SUB_VQ_VP_VR
#undef SUB_VQ_VP_VU

! mma_disassemble_acc
  procedure(sub_atvq) :: __ppc_mma_disassemble_acc
  interface mma_disassemble_acc
    procedure :: __ppc_mma_disassemble_acc
  end interface
  public mma_disassemble_acc

! mma_disassemble_pair
  procedure(sub_atvp) :: __ppc_mma_disassemble_pair
  interface mma_disassemble_pair
    procedure :: __ppc_mma_disassemble_pair
  end interface
  public mma_disassemble_pair

! mma_xxmfacc
  procedure(sub_vq) :: __ppc_mma_xxmfacc
  interface mma_xxmfacc
    procedure :: __ppc_mma_xxmfacc
  end interface
  public mma_xxmfacc

! mma_xxmtacc
  procedure(sub_vq) :: __ppc_mma_xxmtacc
  interface mma_xxmtacc
    procedure :: __ppc_mma_xxmtacc
  end interface
  public mma_xxmtacc

! mma_xxsetaccz
  procedure(sub_vq) :: __ppc_mma_xxsetaccz
  interface mma_xxsetaccz
    procedure :: __ppc_mma_xxsetaccz
  end interface
  public mma_xxsetaccz

end module
