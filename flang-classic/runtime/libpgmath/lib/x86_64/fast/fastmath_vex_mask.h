/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


/*
 *   __fvd_cos_vex_256_mask(argument, mask)
 *   __fvd_cos_fma4_256_mask(argument, mask)
 * 
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the cosine of the arguments whose mask is non-zero
 *
 */
        .text
	ALN_FUNC
	.globl ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256_mask):)

/*	RZ_PUSH */
	subq $8, %rsp

        vptest	.L_zeromask(%rip), %ymm1
	je	LBL(.L_fvd_cos_256_done)

	vandpd	%ymm0,%ymm1,%ymm0
	CALL(ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_cos_256_done):

/*	RZ_POP   */
	addq $8, %rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256_mask))

/*
 *   __fvd_cos_vex_mask(argument, mask)
 *   __fvd_cos_fma4_mask(argument, mask)
 * 
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_cos_done)

        vandpd  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_cos_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_cos_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256_mask):)
        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_cos_256_done)

        vandps %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_cos_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256_mask))

/*
 *   __fvs_cos_vex_mask(argument, mask)
 *   __fvs_cos_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_mask):)
        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_cos_done)

        vandps %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_cos_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvd_sin_vex_256_mask(argument, mask)
 *   __fvd_sin_fma4_256_mask(argument, mask)
 * 
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the sine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256_mask):)
        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je      LBL(.L_fvd_sin_256_done)

        vandpd %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_sin_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256_mask))

/*
 *   __fvd_sin_vex_mask(argument, mask)
 *   __fvd_sin_fma4_mask(argument, mask)
 * 
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the sine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_mask):)
        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_sin_done)

        vandpd %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_sin_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_sin_vex_256_mask(argument, mask)
 *   __fvs_sin_fma4_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the sine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256_mask):)
        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_sin_256_done)

        vandps	%ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_sin_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256_mask))

/*
 *   __fvs_sin_vex_mask(argument, mask)
 *   __fvs_sin_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the sine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_mask):)
        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_sin_done)

        vandps %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_sin_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvd_cosh_vex_256_mask(argument, mask)
 *   __fvd_cosh_fma4_256_mask(argument, mask)
 * 
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the hyperbolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256_mask):)
        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je      LBL(.L_fvd_cosh_256_done)

        vandpd  %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_cosh_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256_mask))

/*
 *   __fvd_cosh_vex_mask(argument, mask)
 *   __fvd_cosh_fma4_mask(argument, mask)
 * 
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the hyperbolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_cosh_done)

        vandpd  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_cosh_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_cosh_vex_256_mask(argument, mask)
 *   __fvs_cosh_fma4_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the hyperbolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_cosh_256_done)

        vandps	%ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_cosh_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_cosh_vex_mask(argument, mask)
 *   __fvs_cosh_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the hyperbolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_cosh_done)

        vandps  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_cosh_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvd_sinh_vex_256_mask(argument, mask)
 *   __fvd_sinh_fma4_256_mask(argument, mask)
 * 
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the hypobolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je      LBL(.L_fvd_sinh_256_done)

        vandpd  %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_sinh_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_sinh_vex_mask(argument, mask)
 *   __fvd_sinh_fma4_mask(argument, mask)
 * 
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the hypobolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_sinh_done)

        vandpd  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_sinh_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvs_sinh_vex_256_mask(argument, mask)
 *   __fvs_sinh_fma4_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the hyperbolic sinine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_sinh_256_done)

        vandps  %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_sinh_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_sinh_vex_mask(argument, mask)
 *   __fvs_sinh_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the hyperbolic sinine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_sinh_done)

        vandps  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_sinh_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvs_sincos_vex_256_mask(argument, mask)
 *   __fvs_sincos_fma4_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the sincos of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

	vmovaps	%ymm1, %ymm3
        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_sincos_256_done)

        vandps  %ymm0,%ymm3,%ymm0
        vandps  %ymm1,%ymm3,%ymm1
        CALL(ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256)))
	addq	$8, %rsp
	ret

LBL(.L_fvs_sincos_256_done):
	vmovaps	%ymm0, %ymm1
        addq	$8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_sincos_vex_mask(argument, mask)
 *   __fvs_sincos_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the sincos of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vmovaps %xmm1, %xmm3
        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_sincos_done)

        vandps  %xmm0,%xmm3,%xmm0
        vandps  %xmm1,%xmm3,%xmm1
        CALL(ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)))
        addq    $8, %rsp
        ret

LBL(.L_fvs_sincos_done):
        vmovaps %xmm0, %xmm1
        addq    $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvd_sincos_vex_256_mask(argument, mask)
 *   __fvd_sincos_fma4_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the hypobolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vmovapd %ymm1, %ymm3
        vptest  .L_zeromask(%rip), %ymm1
        je      LBL(.L_fvd_sincos_256_done)

        vandpd  %ymm0,%ymm3,%ymm0
        vandpd  %ymm1,%ymm3,%ymm1
        CALL(ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256)))
	addq	$8, %rsp
	ret

LBL(.L_fvd_sincos_256_done):
	vmovapd	%ymm0, %ymm1
        addq	$8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_sincos_vex_mask(argument, mask)
 *   __fvd_sincos_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the hypobolic cosine of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vmovapd %xmm1, %xmm3
        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_sincos_done)

        vandpd  %xmm0,%xmm3,%xmm0
        vandpd  %xmm1,%xmm3,%xmm1
        CALL(ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)))
        addq    $8, %rsp
        ret

LBL(.L_fvd_sincos_done):
        vmovapd %xmm0, %xmm1
        addq    $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvd_exp_vex_256_mask(argument, mask)
 *   __fvd_exp_fma4_256_mask(argument, mask)
 * 
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the exp of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je      LBL(.L_fvd_exp_256_done)

        vandpd  %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_exp_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_exp_vex_mask(argument, mask)
 *   __fvd_exp_fma4_mask(argument, mask)
 * 
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the exp of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_exp_done)

        vandpd  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_exp_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvs_exp_vex_256_mask(argument, mask)
 *   __fvs_exp_fma4_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the exp of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_exp_256_done)

        vandps %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_exp_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_exp_vex_mask(argument, mask)
 *   __fvs_exp_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the exp of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_exp_done)

        vandps %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_exp_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvd_pow_vex_256_mask(argument1, argument2, mask)
 *   __fvd_pow_fma4_256_mask(argument1, argument2, mask)
 * 
 *   argument:   ymm0, ymm1
 *   mask:       ymm2
 *
 *   Compute the power of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm2
        je      LBL(.L_fvd_pow_256_done)

        vmovupd .L_dpow_mask_two(%rip),%ymm3
        vblendvpd %ymm2,%ymm0,%ymm3,%ymm0
        vblendvpd %ymm2,%ymm1,%ymm3,%ymm1


        CALL(ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_pow_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_pow_vex_mask(argument1, argument2, mask)
 *   __fvd_pow_fma4_mask(argument1, argument2, mask)
 * 
 *   argument:   xmm0, xmm1
 *   mask:       xmm2
 *
 *   Compute the power of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm2
        je      LBL(.L_fvd_pow_done)

        vmovupd .L_dpow_mask_two(%rip),%xmm3
        vblendvpd %xmm2,%xmm0,%xmm3,%xmm0
        vblendvpd %xmm2,%xmm1,%xmm3,%xmm1

        CALL(ENT(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_pow_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvs_pow_vex_256_mask(argument1, argument2, mask)
 *   __fvs_pow_fma4_256_mask(argument1, argument2, mask)
 *
 *   argument:   ymm0, ymm1
 *   mask:       ymm2
 *
 *   Compute the power of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm2
        je      LBL(.L_fvs_pow_256_done)

        vmovups .L_spow_mask_two(%rip),%ymm3
        vblendvps %ymm2,%ymm0,%ymm3,%ymm0
        vblendvps %ymm2,%ymm1,%ymm3,%ymm1

        CALL(ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_pow_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_pow_vex_mask(argument1, argument2, mask)
 *   __fvs_pow_fma4_mask(argument1, argument2, mask)
 *
 *   argument:   xmm0, xmm1
 *   mask:       xmm2
 *
 *   Compute the power of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm2
        je      LBL(.L_fvs_pow_done)

        vmovups .L_spow_mask_two(%rip),%xmm3
        vblendvps %xmm2,%xmm0,%xmm3,%xmm0
        vblendvps %xmm2,%xmm1,%xmm3,%xmm1


        CALL(ENT(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_pow_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_sqrt_fma4_256_mask(argument, mask)
 *   __fvs_sqrt_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the square root of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je LBL(.L_done_fvs_sqrt_256)

        vandps %ymm0,%ymm1,%ymm0
        vsqrtps %ymm0,%ymm0

LBL(.L_done_fvs_sqrt_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_sqrt_fma4_mask(argument, mask)
 *   __fvs_sqrt_vex_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the square root of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je LBL(.L_done_fvs_sqrt)

        vandps %xmm0,%xmm1,%xmm0
        vsqrtps %xmm0,%xmm0

LBL(.L_done_fvs_sqrt):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_sqrt_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvd_sqrt_fma4_256_mask(argument, mask)
 *   __fvd_sqrt_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the square root of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je LBL(.L_done_fvd_sqrt_256)

        vandpd %ymm0,%ymm1,%ymm0
        vsqrtpd %ymm0,%ymm0

LBL(.L_done_fvd_sqrt_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_sqrt_fma4_mask(argument, mask)
 *   __fvd_sqrt_vex_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the square root of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je LBL(.L_done_fvd_sqrt)

        vandpd %xmm0,%xmm1,%xmm0
        vsqrtpd %xmm0,%xmm0

LBL(.L_done_fvd_sqrt):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_sqrt_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_div_fma4_256_mask(argument1, argument2, mask)
 *   __fvs_div_vex_256_mask(argument1, argument2, mask)
 *
 *   argument1(dividend):   ymm0
 *   argument2(divisor):    ymm1
 *   mask:                  ymm2
 *
 *   Compute argument1 / argument2 whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm2
        je LBL(.L_done_fvs_div_256)

        vmovups .L_one_for_mask_fvs(%rip), %ymm3

        vblendvps %ymm2,%ymm0,%ymm3,%ymm0
        vblendvps %ymm2,%ymm1,%ymm3,%ymm1

        vdivps %ymm1,%ymm0,%ymm0

LBL(.L_done_fvs_div_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_div_fma4_mask(argument1, argument2, mask)
 *   __fvs_div_vex_mask(argument1, argument2, mask)
 *
 *   argument1(dividend):   xmm0
 *   argument2(divisor):    xmm1
 *   mask:                  xmm2
 *
 *   Compute argument1 / argument2 whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm2
        je LBL(.L_done_fvs_div)

        vmovups .L_one_for_mask_fvs(%rip), %xmm3

        vblendvps %xmm2,%xmm0,%xmm3,%xmm0
        vblendvps %xmm2,%xmm1,%xmm3,%xmm1

        vdivps %xmm1,%xmm0,%xmm0

LBL(.L_done_fvs_div):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_div_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvd_div_fma4_256_mask(argument1, argument2, mask)
 *   __fvd_div_vex_256_mask(argument1, argument2, mask)
 *
 *   argument1(dividend):   ymm0
 *   argument2(divisor):    ymm1
 *   mask:                  ymm2
 *
 *   Compute argument1 / argument2 whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm2
        je LBL(.L_done_fvd_div_256)

        vmovupd .L_one_for_mask_fvd(%rip), %ymm3

        vblendvpd %ymm2,%ymm0,%ymm3,%ymm0
        vblendvpd %ymm2,%ymm1,%ymm3,%ymm1

        vdivpd %ymm1,%ymm0,%ymm0

LBL(.L_done_fvd_div_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_div_fma4_mask(argument1, argument2, mask)
 *   __fvd_div_vex_mask(argument1, argument2, mask)
 *
 *   argument1(dividend):   xmm0
 *   argument2(divisor):    xmm1
 *   mask:                  xmm2
 *
 *   Compute argument1 / argument2 whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm2
        je LBL(.L_done_fvd_div)

        vmovupd .L_one_for_mask_fvd(%rip), %xmm3

        vblendvpd %xmm2,%xmm0,%xmm3,%xmm0
        vblendvpd %xmm2,%xmm1,%xmm3,%xmm1

        vdivpd %xmm1,%xmm0,%xmm0

LBL(.L_done_fvd_div):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_div_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_log_fma4_256_mask(argument, mask)
 *   __fvs_log_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the logarithm of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je LBL(.L_done_fvs_log_256)

        vmovups .L_one_for_mask_fvs(%rip), %ymm2

        vblendvps %ymm1,%ymm0,%ymm2,%ymm0

        CALL(ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_done_fvs_log_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_log_fma4_mask(argument, mask)
 *   __fvs_log_vex_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the logarithm of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je LBL(.L_done_fvs_log)

        vmovups .L_one_for_mask_fvs(%rip), %xmm2

        vblendvps %xmm1,%xmm0,%xmm2,%xmm0

        CALL(ENT(ASM_CONCAT(__fvs_log_,TARGET_VEX_OR_FMA)))

LBL(.L_done_fvs_log):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvd_log_fma4_256_mask(argument, mask)
 *   __fvd_log_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the logarithm of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je LBL(.L_done_fvd_log_256)

        vmovupd .L_one_for_mask_fvd(%rip), %ymm2

        vblendvpd %ymm1,%ymm0,%ymm2,%ymm0

        CALL(ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_done_fvd_log_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_log_fma4_mask(argument, mask)
 *   __fvd_log_vex_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the logarithm of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je LBL(.L_done_fvd_log)

        vmovupd .L_one_for_mask_fvd(%rip), %xmm2

        vblendvpd %xmm1,%xmm0,%xmm2,%xmm0

        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))

LBL(.L_done_fvd_log):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvs_log10_fma4_256_mask(argument, mask)
 *   __fvs_log10_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the logarithm(base 10) of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je LBL(.L_done_fvs_log10_256)

        vmovups .L_one_for_mask_fvs(%rip), %ymm2

        vblendvps %ymm1,%ymm0,%ymm2,%ymm0

        CALL(ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_done_fvs_log10_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_log10_fma4_mask(argument, mask)
 *   __fvs_log10_vex_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the logarithm(base 10) of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je LBL(.L_done_fvs_log10)

        vmovups .L_one_for_mask_fvs(%rip), %xmm2

        vblendvps %xmm1,%xmm0,%xmm2,%xmm0

        CALL(ENT(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA)))

LBL(.L_done_fvs_log10):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_mask))


/*
 *   __fvd_log10_fma4_256_mask(argument, mask)
 *   __fvd_log10_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the logarithm(base 10) of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %ymm1
        je LBL(.L_done_fvd_log10_256)

        vmovupd .L_one_for_mask_fvd(%rip), %ymm2

        vblendvpd %ymm1,%ymm0,%ymm2,%ymm0

        CALL(ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_done_fvd_log10_256):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256_mask))



/*
 *   __fvd_log10_fma4_mask(argument, mask)
 *   __fvd_log10_vex_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the logarithm(base 10) of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je LBL(.L_done_fvd_log10)

        vmovupd .L_one_for_mask_fvd(%rip), %xmm2

        vblendvpd %xmm1,%xmm0,%xmm2,%xmm0

        CALL(ENT(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA)))

LBL(.L_done_fvd_log10):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvd_tan_vex_256_mask(argument, mask)
 *   __fvd_tan_fma4_256_mask(argument, mask)
 * 
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the tangent of the arguments whose mask is non-zero
 *
 */
        .text
	ALN_FUNC
	.globl ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256_mask):)


	subq $8, %rsp

        vptest	.L_zeromask(%rip), %ymm1
	je	LBL(.L_fvd_tan_256_done)

	vandpd	%ymm0,%ymm1,%ymm0
	CALL(ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvd_tan_256_done):

	addq $8, %rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvd_tan_vex_mask(argument, mask)
 *   __fvd_tan_fma4_mask(argument, mask)
 * 
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the tangent of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_zeromask(%rip), %xmm1
        je      LBL(.L_fvd_tan_done)

        vandpd  %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA)))

LBL(.L_fvd_tan_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_mask))



/*
 *   __fvs_tan_vex_256_mask(argument, mask)
 *
 *   argument:   ymm0
 *   mask:       ymm1
 *
 *   Compute the tangent of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256_mask))
ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %ymm1
        je      LBL(.L_fvs_tan_256_done)

        vandps %ymm0,%ymm1,%ymm0
        CALL(ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256)))

LBL(.L_fvs_tan_256_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256_mask))


/*
 *   __fvs_tan_vex_mask(argument, mask)
 *   __fvs_tan_fma4_mask(argument, mask)
 *
 *   argument:   xmm0
 *   mask:       xmm1
 *
 *   Compute the tangent of the arguments whose mask is non-zero
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_mask))
ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_mask):)

        subq $8, %rsp

        vptest  .L_s_zeromask(%rip), %xmm1
        je      LBL(.L_fvs_tan_done)

        vandps %xmm0,%xmm1,%xmm0
        CALL(ENT(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA)))

LBL(.L_fvs_tan_done):
        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_mask))
        ELF_SIZE(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_mask))


