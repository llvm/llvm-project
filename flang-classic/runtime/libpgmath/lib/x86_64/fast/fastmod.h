/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * ============================================================
 */


#ifndef LNUM
#define LNUM 001
#endif
#define NNN LNUM

#ifdef TABLE_TARGET

        ALN_QUAD
.L_zeromask_mod:
        .quad 0xFFFFFFFFFFFFFFFF
        .quad 0xFFFFFFFFFFFFFFFF
        .quad 0xFFFFFFFFFFFFFFFF
        .quad 0xFFFFFFFFFFFFFFFF

        ALN_QUAD
.L_one_mod_mask_fvs_256:
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */

        ALN_QUAD
.L_one_mod_mask_fvs:
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */

        ALN_QUAD
.L_one_mod_mask_fvd_256:
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */

        ALN_QUAD
.L_one_mod_mask_fvd:
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */


#else


#ifdef TARGET_VEX_OR_FMA

#include "fastmod_vex.h"
#include "fastmod_vex_mask.h"

#else


	.text
	ALN_FUNC
#ifdef GH_TARGET
	.globl ENT(__fvs_mod)
	.globl ENT(__fvsmod_gh)
ENT(__fvs_mod):
ENT(__fvsmod_gh):
#else
	.globl ENT(__fvsmod)
ENT(__fvsmod):
#endif
        RZ_PUSH

        /* Move all data to memory, then 1st piece to fp stack */
        movaps  %xmm1, RZ_OFF(40)(%rsp)
        movaps  %xmm0, RZ_OFF(24)(%rsp)
        flds      RZ_OFF(40)(%rsp)
        flds      RZ_OFF(24)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlps1):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlps1)

        /* Store result back to memory */
        fstps     RZ_OFF(24)(%rsp)
        fstp      %st(0)

        /* 2 */
        flds      RZ_OFF(36)(%rsp)
        flds      RZ_OFF(20)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlps2):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlps2)

        /* Store result back to memory */
        fstps     RZ_OFF(20)(%rsp)
        fstp      %st(0)

        /* 3 */
        flds      RZ_OFF(32)(%rsp)
        flds      RZ_OFF(16)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlps3):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlps3)

        /* Store result back to memory */
        fstps     RZ_OFF(16)(%rsp)
        fstp      %st(0)

        /* 4 */
        flds      RZ_OFF(28)(%rsp)
        flds      RZ_OFF(12)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlps4):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlps4)

        /* Store result back to memory */
        fstps     RZ_OFF(12)(%rsp)
        fstp      %st(0)

        /* Store back to xmm0 */
        movaps    RZ_OFF(24)(%rsp), %xmm0
        RZ_POP
        ret

#ifdef GH_TARGET
        ELF_FUNC(__fvsmod_gh)
        ELF_SIZE(__fvsmod_gh)
        ELF_FUNC(__fvs_mod)
        ELF_SIZE(__fvs_mod)
#else
        ELF_FUNC(__fvsmod)
        ELF_SIZE(__fvsmod)
#endif

/* ========================================================================= */

	.text
	ALN_FUNC
#ifdef GH_TARGET
	.globl ENT(__fvd_mod)
	.globl ENT(__fvdmod_gh)
ENT(__fvd_mod):
ENT(__fvdmod_gh):
#else
	.globl ENT(__fvdmod)
ENT(__fvdmod):
#endif
        RZ_PUSH

        /* Move all data to memory, then 1st piece to fp stack */
        movapd  %xmm1, RZ_OFF(40)(%rsp)
        movapd  %xmm0, RZ_OFF(24)(%rsp)
        fldl      RZ_OFF(40)(%rsp)
        fldl      RZ_OFF(24)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlpd1):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlpd1)

        /* Store result back to memory */
        fstpl     RZ_OFF(24)(%rsp)
        fstp      %st(0)

        fldl      RZ_OFF(32)(%rsp)
        fldl      RZ_OFF(16)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlpd2):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlpd2)

        /* Store result back to memory, then xmm0 */
        fstpl     RZ_OFF(16)(%rsp)
        fstp      %st(0)
        movapd    RZ_OFF(24)(%rsp), %xmm0

        RZ_POP
	ret

#ifdef GH_TARGET
        ELF_FUNC(__fvdmod_gh)
        ELF_SIZE(__fvdmod_gh)
        ELF_FUNC(__fvd_mod)
        ELF_SIZE(__fvd_mod)
#else
        ELF_FUNC(__fvdmod)
        ELF_SIZE(__fvdmod)
#endif

/* ========================================================================= */

	.text
        ALN_FUNC
#ifdef GH_TARGET
	.globl ENT(__fsd_mod)
	.globl ENT(__fmth_i_dmod_gh)
ENT(__fsd_mod):
ENT(__fmth_i_dmod_gh):
#else
	.globl ENT(__fmth_i_dmod)
ENT(__fmth_i_dmod):
#ifdef	TARGET_WIN_X8664
	.globl ENT(__mth_i_dmod)
ENT(__mth_i_dmod):
#endif
#endif
	RZ_PUSH

        /* Move arguments to fp stack */
        movsd     %xmm1, RZ_OFF(24)(%rsp)
        movsd     %xmm0, RZ_OFF(16)(%rsp)
        fldl      RZ_OFF(24)(%rsp)
        fldl      RZ_OFF(16)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlpd):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlpd)

        /* Store result back to xmm0 */
        fstpl     RZ_OFF(16)(%rsp)
        fstp      %st(0)
        movsd     RZ_OFF(16)(%rsp), %xmm0
        RZ_POP
        ret

#ifdef GH_TARGET
        ELF_FUNC(__fmth_i_dmod_gh)
        ELF_SIZE(__fmth_i_dmod_gh)
        ELF_FUNC(__fsd_mod)
        ELF_SIZE(__fsd_mod)
#else
	ELF_FUNC(__fmth_i_dmod)
	ELF_SIZE(__fmth_i_dmod)
#ifdef	TARGET_WIN_X8664
	ELF_FUNC(__mth_i_dmod)
	ELF_SIZE(__mth_i_dmod)
#endif
#endif

/* ========================================================================= */

	.text
        ALN_FUNC
#ifdef GH_TARGET
	.globl ENT(__fss_mod)
	.globl ENT(__fmth_i_amod_gh)
ENT(__fss_mod):
ENT(__fmth_i_amod_gh):
#else
	.globl ENT(__fmth_i_amod)
ENT(__fmth_i_amod):
#ifdef	TARGET_WIN_X8664
	.globl ENT(__mth_i_amod)
ENT(__mth_i_amod):
#endif
#endif
	RZ_PUSH

        /* Move arguments to fp stack */
        movss     %xmm1, RZ_OFF(12)(%rsp)
        movss     %xmm0, RZ_OFF(8)(%rsp)
        flds      RZ_OFF(12)(%rsp)
        flds      RZ_OFF(8)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlps):
        fprem
        fstsw     %ax
        test      $4, %ah
        jnz       LBL(.L_remlps)

        /* Store result back to xmm0 */
        fstps     RZ_OFF(8)(%rsp)
        fstp      %st(0)
        movss     RZ_OFF(8)(%rsp), %xmm0
        RZ_POP
        ret

#ifdef GH_TARGET
        ELF_FUNC(__fmth_i_amod_gh)
        ELF_SIZE(__fmth_i_amod_gh)
        ELF_FUNC(__fss_mod)
        ELF_SIZE(__fss_mod)
#else
	ELF_FUNC(__fmth_i_amod)
	ELF_SIZE(__fmth_i_amod)
#ifdef	TARGET_WIN_X8664
	ELF_FUNC(__mth_i_amod)
	ELF_SIZE(__mth_i_amod)
#endif
#endif

#endif

#endif
