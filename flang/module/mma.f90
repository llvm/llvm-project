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

!! ========== 3 arguments subroutine interface ===============================!!
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

#undef ELEM_SUB_VPVIVI
#undef ELEM_SUB_VPVUVU
#undef ELEM_SUB_VPVRVR

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

end module

