/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */



	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA)):

        RZ_PUSH

        /* Move all data to memory, then 1st piece to fp stack */
        vmovaps  %xmm1, RZ_OFF(40)(%rsp)
        vmovaps  %xmm0, RZ_OFF(24)(%rsp)
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
        vmovaps    RZ_OFF(24)(%rsp), %xmm0
        RZ_POP
        ret

        ELF_FUNC(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA))

/* ========================================================================= */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA)):

        RZ_PUSH

        /* Move all data to memory, then 1st piece to fp stack */
        vmovapd  %xmm1, RZ_OFF(40)(%rsp)
        vmovapd  %xmm0, RZ_OFF(24)(%rsp)
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
        vmovapd    RZ_OFF(24)(%rsp), %xmm0

        RZ_POP
	ret

        ELF_FUNC(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA))

/* ========================================================================= */

	.text
        ALN_FUNC
	.globl ENT(ASM_CONCAT(__fsd_mod_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_mod_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        /* Move arguments to fp stack */
        vmovsd     %xmm1, RZ_OFF(24)(%rsp)
        vmovsd     %xmm0, RZ_OFF(16)(%rsp)
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
        vmovsd     RZ_OFF(16)(%rsp), %xmm0
        RZ_POP
        ret

	ELF_FUNC(ASM_CONCAT(__fsd_mod_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fsd_mod_,TARGET_VEX_OR_FMA))

/* ========================================================================= */

	.text
        ALN_FUNC
	.globl ENT(ASM_CONCAT(__fss_mod_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_mod_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        /* Move arguments to fp stack */
        vmovss     %xmm1, RZ_OFF(12)(%rsp)
        vmovss     %xmm0, RZ_OFF(8)(%rsp)
        flds      RZ_OFF(12)(%rsp)
        flds      RZ_OFF(8)(%rsp)

        /* Loop over partial remainder until done */
LBL(.L_remlps):
        fprem
        fstsw	%ax
        test	$4, %ah
        jnz	LBL(.L_remlps)

        /* Store result back to xmm0 */
        fstps	RZ_OFF(8)(%rsp)
        fstp	%st(0)
        vmovss	RZ_OFF(8)(%rsp), %xmm0
        RZ_POP
        ret

	ELF_FUNC(ASM_CONCAT(__fss_mod_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_mod_,TARGET_VEX_OR_FMA))

/* ------------------------------------------------------------------------- */
/* 
 *  vector sinle precision mod
 *
 *  Prototype:
 *
 *      single __fvs_mod_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256)):

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 32(%rsp)
        vmovups %ymm1, 96(%rsp)
        CALL(ENT(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA)))

        vmovups         32(%rsp), %ymm2
        vmovups         96(%rsp), %ymm4
        vmovaps         %xmm0, %xmm3
        vextractf128    $1, %ymm2, %xmm2
        vextractf128    $1, %ymm4, %xmm4
        vmovaps         %xmm2, %xmm0
        vmovaps         %xmm4, %xmm1
        vmovups         %ymm3, 64(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA)))
        vmovups 64(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256))


/* ------------------------------------------------------------------------- */
/* 
 *  vector double precision mod
 * 
 *  Prototype:
 * 
 *      double __fvd_mod_vex/fma4_256(double *x);
 * 
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256)):

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 32(%rsp)
        vmovups %ymm1, 96(%rsp)
        CALL(ENT(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA)))

        vmovups         32(%rsp), %ymm2
        vmovups         96(%rsp), %ymm4
        vmovaps         %xmm0, %xmm3
        vextractf128    $1, %ymm2, %xmm2
        vextractf128    $1, %ymm4, %xmm4
        vmovaps         %xmm2, %xmm0
        vmovaps         %xmm4, %xmm1
        vmovups         %ymm3, 64(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA)))
        vmovups 64(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256))
