/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */



/*
 *   __fvs_mod_fma4_256_mask(argument1, argument2, mask)
 *   __fvs_mod_vex_256_mask(argument1, argument2, mask)
 * 
 *   argument1(dividend):   ymm0
 *   argument2(divisor):    ymm1
 *   mask:                  ymm2
 *
 *   Compute mod(argument1,argument2) whose mask is non-zero
 *
 */
        .text
	ALN_FUNC
	.globl ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256_mask)):

	RZ_PUSH
	subq $8, %rsp

        vptest	.L_zeromask_mod(%rip), %ymm2
	je LBL(.L_done_fvs_mod_256)

	vmovups .L_one_mod_mask_fvs_256(%rip), %ymm3

	vblendvps %ymm2,%ymm0,%ymm3,%ymm0
	vblendvps %ymm2,%ymm1,%ymm3,%ymm1

        CALL(ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_done_fvs_mod_256):
	RZ_POP
	addq $8, %rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_mod_fma4_mask(argument1, argument2, mask)
 *   __fvs_mod_vex_mask(argument1, argument2, mask)
 * 
 *   argument1(dividend):   xmm0
 *   argument2(divisor):    xmm1
 *   mask:                  xmm2
 *
 *   Compute mod(argument1,argument2) whose mask is non-zero
 *
 */
        .text
	ALN_FUNC
	.globl ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_mask)):

	RZ_PUSH
	subq $8, %rsp

        vptest	.L_zeromask_mod(%rip), %xmm2
	je LBL(.L_done_fvs_mod)

	vmovups .L_one_mod_mask_fvs(%rip), %xmm3

	vblendvps %xmm2,%xmm0,%xmm3,%xmm0
	vblendvps %xmm2,%xmm1,%xmm3,%xmm1

        CALL(ENT(ASM_CONCAT(__fvs_mod_,TARGET_VEX_OR_FMA)))

LBL(.L_done_fvs_mod):
	RZ_POP
	addq $8, %rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_mod_,TARGET_VEX_OR_FMA,_mask))

/*
 *   __fvd_mod_fma4_256_mask(argument1, argument2, mask)
 *   __fvd_mod_vex_256_mask(argument1, argument2, mask)
 * 
 *   argument1(dividend):   ymm0
 *   argument2(divisor):    ymm1
 *   mask:                  ymm2
 *
 *   Compute mod(argument1,argument2) whose mask is non-zero
 *
 */
        .text
	ALN_FUNC
	.globl ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256_mask)):

	RZ_PUSH
	subq $8, %rsp

        vptest	.L_zeromask_mod(%rip), %ymm2
	je LBL(.L_done_fvd_mod_256)

	vmovupd .L_one_mod_mask_fvd_256(%rip), %ymm3

	vblendvpd %ymm2,%ymm0,%ymm3,%ymm0
	vblendvpd %ymm2,%ymm1,%ymm3,%ymm1

        CALL(ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_done_fvd_mod_256):
	RZ_POP
	addq $8, %rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_mod_fma4_mask(argument1, argument2, mask)
 *   __fvd_mod_vex_mask(argument1, argument2, mask)
 * 
 *   argument1(dividend):   xmm0
 *   argument2(divisor):    xmm1
 *   mask:                  xmm2
 *
 *   Compute mod(argument1,argument2) whose mask is non-zero
 *
 */
        .text
	ALN_FUNC
	.globl ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_mask)):

	RZ_PUSH
	subq $8, %rsp

        vptest	.L_zeromask_mod(%rip), %xmm2
	je LBL(.L_done_fvd_mod)

	vmovupd .L_one_mod_mask_fvd(%rip), %xmm3

	vblendvpd %xmm2,%xmm0,%xmm3,%xmm0
	vblendvpd %xmm2,%xmm1,%xmm3,%xmm1

        CALL(ENT(ASM_CONCAT(__fvd_mod_,TARGET_VEX_OR_FMA)))

LBL(.L_done_fvd_mod):
	RZ_POP
	addq $8, %rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_mod_,TARGET_VEX_OR_FMA,_mask))
