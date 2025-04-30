
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* ============================================================ */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__rvs_pow_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rvs_pow_,TARGET_VEX_OR_FMA)):

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
	vmovaps	%xmm0, %xmm4
	vmovaps	%xmm1, %xmm5
        vmovaps  %xmm0, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3
	vandps	.L4_fvspow_infinity_mask(%rip), %xmm4, %xmm4
	vandps	.L4_fvspow_infinity_mask(%rip), %xmm5, %xmm5
	vcmpps	$2, %xmm3, %xmm2, %xmm2
	vcmpps	$0, .L4_fvspow_infinity_mask(%rip), %xmm4, %xmm4
	vcmpps	$0, .L4_fvspow_infinity_mask(%rip), %xmm5, %xmm5
	vorps	%xmm4, %xmm2, %xmm2
	/* Store input arguments onto stack */
        vmovaps  %xmm0, _SX0(%rsp) 
	vorps	%xmm5, %xmm2, %xmm2
        vmovaps  %xmm1, _SY0(%rsp)
	vmovmskps %xmm2, %r8d
	test	$15, %r8d
	jnz	LBL(.L__Scalar_fvspow)

	/* Convert x0, x1 to dbl and call log */
/*	vcvtps2pd %xmm0, %xmm0
        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))

*/

        CALL(ENT(ASM_CONCAT(__fvs_log_,TARGET_VEX_OR_FMA)))


	/* dble(y) * dlog(x) */
/*        vmovlps  _SY0(%rsp), %xmm1, %xmm1 */
        vmovaps  _SY0(%rsp), %xmm1
/*	vcvtps2pd %xmm1, %xmm1 */
/*	vmulpd	%xmm1, %xmm0, %xmm0 */
	vmulps	%xmm1, %xmm0, %xmm0
/*        vmovapd  %xmm0, _SR0(%rsp) */

	/* Convert x2, x3 to dbl and call log */
/*        vmovlps  _SX2(%rsp), %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm0
        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))

*/

	/* dble(y) * dlog(x) */
/*        vmovlps  _SY2(%rsp), %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vmulpd	%xmm0, %xmm1, %xmm1
	vmovapd	_SR0(%rsp), %xmm0
	CALL(ENT(ASM_CONCAT(__fvs_exp_dbl_,TARGET_VEX_OR_FMA)))

*/

        CALL(ENT(ASM_CONCAT(__rvs_exp_,TARGET_VEX_OR_FMA)))


        movq    %rbp, %rsp
        popq    %rbp
	ret

LBL(.L__Scalar_fvspow):
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovss   _SX1(%rsp), %xmm0
        vmovss   _SY1(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovss   _SX2(%rsp), %xmm0
        vmovss   _SY2(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovss   _SX3(%rsp), %xmm0
        vmovss   _SY3(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT(__rvs_pow_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rvs_pow_,TARGET_VEX_OR_FMA))


/* ========================================================================= */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)):


	vmovaps	%xmm1, %xmm2
	vmovaps	%xmm1, %xmm3
	vmovaps	%xmm1, %xmm4
	vshufps	$0, %xmm0, %xmm2, %xmm2
	vshufps	$0, %xmm0, %xmm3, %xmm3
	vshufps	$0, %xmm0, %xmm4, %xmm4
	vmovd	%xmm0, %eax
	vmovd	%xmm1, %ecx
	vcmpps	$0, .L4_100(%rip), %xmm2, %xmm2
	vcmpps	$0, .L4_101(%rip), %xmm3, %xmm3
	vandps	.L4_102(%rip), %xmm4, %xmm4
	vmovdqa	%xmm4, %xmm5
	vorps	%xmm3, %xmm2, %xmm2
	vpcmpeqd	.L4_103(%rip), %xmm5, %xmm5
	vorps	%xmm5, %xmm2, %xmm2
	vmovmskps %xmm2, %r8d
	test	$15, %r8d
	jnz	LBL(.L__Special_Pow_Cases)
	vcomiss	.L4_104(%rip), %xmm4
	ja	LBL(.L__Y_is_large)
	vcomiss	.L4_105(%rip), %xmm4
	jb	LBL(.L__Y_near_zero)
/*	vunpcklps %xmm1, %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm0 */
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$128, %rsp
/*	vmovsd	%xmm1, 0(%rsp) */
	vmovss	%xmm1, 0(%rsp)
/* #ifdef TARGET_FMA
	CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))
 */

	CALL(ENT(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA)))


/* 	vmulsd	0(%rsp), %xmm0, %xmm0 */
	vmulss	0(%rsp), %xmm0, %xmm0

/* #ifdef TARGET_FMA
	CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))
 */

	CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

	movq	%rbp, %rsp
	popq	%rbp
	ret

LBL(.L__Special_Pow_Cases):
	/* if x == 1.0, return 1.0 */
	cmp	.L4_101(%rip), %eax
	je	LBL(.L__Special_Case_1)

	/* if y == 1.5, return x * sqrt(x) */
	cmp	.L4_100+4(%rip), %ecx
	je	LBL(.L__Special_Case_2)

	/* if y == 0.5, return sqrt(x) */
	cmp	.L4_100(%rip), %ecx
	je	LBL(.L__Special_Case_3)

	/* if y == 0.25, return sqrt(sqrt(x)) */
	cmp	.L4_101+4(%rip), %ecx
	je	LBL(.L__Special_Case_4)

	/* if abs(y) == 0, return 1.0 */
	test	.L4_102(%rip), %ecx
	je	LBL(.L__Special_Case_5)

	/* if x == nan or inf, handle */
	mov	%eax, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_6)

LBL(.L__Special_Pow_Case_7):
	/* if y == nan or inf, handle */
	mov	%ecx, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_7)

LBL(.L__Special_Pow_Case_8):
	/* if y == 1.0, return x */
	cmp	.L4_101(%rip), %ecx
	jne	LBL(.L__Special_Pow_Case_9)
	rep
	ret

LBL(.L__Special_Pow_Case_9):
	/* If sign of x is 1, jump away */
	test	.L4_102+8(%rip), %eax
	jne	LBL(.L__Special_Pow_Case_10)
	/* x is 0.0, pos, or +inf */
	mov	%eax, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_9b)
LBL(.L__Special_Case_9a):
	/* x is 0.0, test sign of y */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_102+4(%rip), %eax
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_9b):
	/* x is +inf, test sign of y */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_100+12(%rip), %eax
	vmovd	%eax, %xmm0
	ret
	
LBL(.L__Special_Pow_Case_10):
	/* x is -0.0, neg, or -inf */
	/* Need to compute y is integer, even, odd, etc. */
	mov	%ecx, %r8d
	mov	%ecx, %r9d
	mov	$150, %r10d
	and	.L4_102+4(%rip), %r8d
	sar	$23, %r8d
	sub	%r8d, %r10d 	/* 150 - ((y && 0x7f8) >> 23) */
	jb	LBL(.L__Y_inty_2)
	cmp	$24, %r10d
	jae	LBL(.L__Y_inty_0)
	mov	$1, %edx
	mov	%r10d, %ecx
	shl	%cl, %edx
	mov	%edx, %r10d
	sub	$1, %edx
	test	%r9d, %edx
	jne	LBL(.L__Y_inty_0)
	test	%r9d, %r10d
	jne	LBL(.L__Y_inty_1)
LBL(.L__Y_inty_2):
	mov	$2, %r8d
	jmp	LBL(.L__Y_inty_decided)
LBL(.L__Y_inty_1):
	mov	$1, %r8d
	jmp	LBL(.L__Y_inty_decided)
LBL(.L__Y_inty_0):
	xor	%r8d, %r8d

LBL(.L__Y_inty_decided):
	mov	%r9d, %ecx
	mov	%eax, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_10c)
LBL(.L__Special_Case_10a):
	test	.L4_102(%rip), %eax
	jne	LBL(.L__Special_Case_10e)
	/* x is -0.0, test sign of y */
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10b)
	xor	%eax, %eax
	test	.L4_102+8(%rip), %ecx
	cmovne 	.L4_102+4(%rip), %eax
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10b):
	test	.L4_102+8(%rip), %ecx
	cmovne 	.L4_108(%rip), %eax
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10c):
	/* x is -inf, test sign of y */
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10d)
	/* x is -inf, inty != 1 */
	mov	.L4_102+4(%rip), %eax
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_100+12(%rip), %eax
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10d):
	/* x is -inf, inty == 1 */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_102+8(%rip), %eax
	vmovd	%eax, %xmm0
	ret

LBL(.L__Special_Case_10e):
	/* x is negative */
	vcomiss	.L4_104(%rip), %xmm4
	ja	LBL(.L__Y_is_large)
	test	$3, %r8d
	je	LBL(.L__Special_Case_10f)
	and	.L4_102(%rip), %eax
	vmovd	%eax, %xmm0
/*	vunpcklps %xmm1, %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm0 */
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$128, %rsp
/*	vmovsd	%xmm1, 0(%rsp) */
	vmovss	%xmm1, 0(%rsp)
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10g)

/* #ifdef TARGET_FMA
	CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))
 */

	CALL(ENT(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA)))

/*	vmulsd	0(%rsp), %xmm0, %xmm0 */
	vmulss	0(%rsp), %xmm0, %xmm0

/* #ifdef TARGET_FMA
	CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))
 */

	CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

	movq	%rbp, %rsp
	popq	%rbp
	ret

LBL(.L__Special_Case_10f):
/*
	and	.L4_101+12(%rip), %eax
	or 	.L4_102+12(%rip), %eax
*/
/* Changing this on Sept 13, 2005, to return 0xffc00000 for neg ** neg */
	mov 	.L4_108(%rip), %eax
	or 	.L4_107(%rip), %eax
	vmovd	%eax, %xmm0
	ret

LBL(.L__Special_Case_10g):
/* #ifdef TARGET_FMA
	CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))

	vmulsd	0(%rsp), %xmm0, %xmm0
	CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))
 */

	CALL(ENT(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA)))

	vmulss	0(%rsp), %xmm0, %xmm0
	CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

	vmovaps	%xmm0, %xmm1
	vxorps	%xmm0, %xmm0, %xmm0
	vsubps	%xmm1, %xmm0, %xmm0
	movq	%rbp, %rsp
	popq	%rbp
	ret


LBL(.L__Special_Case_1):
LBL(.L__Special_Case_5):
	vmovss	.L4_101(%rip), %xmm0
	ret
LBL(.L__Special_Case_2):
	vsqrtss	%xmm0, %xmm1, %xmm1
	vmulss	%xmm1, %xmm0, %xmm0
	ret
LBL(.L__Special_Case_3):
	vsqrtss	%xmm0, %xmm0, %xmm0
	ret
LBL(.L__Special_Case_4):
	vsqrtss	%xmm0, %xmm0, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm0
	ret
LBL(.L__Special_Case_6):
	test	.L4_106(%rip), %eax
	je	LBL(.L__Special_Pow_Case_7)
	or	.L4_107(%rip), %eax
	vmovd	%eax, %xmm0
	ret

LBL(.L__Special_Case_7):
	test	.L4_106(%rip), %ecx
	je	LBL(.L__Y_is_large)
	or	.L4_107(%rip), %ecx
	vmovd	%ecx, %xmm0
	ret

/* This takes care of all the large Y cases */
LBL(.L__Y_is_large):
	vcomiss	.L4_103(%rip), %xmm1
	vandps	.L4_102(%rip), %xmm0, %xmm0
	jb	LBL(.L__Y_large_negative)
LBL(.L__Y_large_positive):
	/* If abs(x) < 1.0, return 0 */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return Inf */
	vcomiss	.L4_101(%rip), %xmm0
	jb	LBL(.L__Y_large_pos_0)
	je	LBL(.L__Y_large_pos_1)
LBL(.L__Y_large_pos_i):
	vmovss	.L4_102+4(%rip), %xmm0
	ret
LBL(.L__Y_large_pos_1):
	vmovss	.L4_101(%rip), %xmm0
	ret
/* */
LBL(.L__Y_large_negative):
	/* If abs(x) < 1.0, return Inf */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return 0 */
	vcomiss	.L4_101(%rip), %xmm0
	jb	LBL(.L__Y_large_pos_i)
	je	LBL(.L__Y_large_pos_1)
LBL(.L__Y_large_pos_0):
	vmovss	.L4_103(%rip), %xmm0
	ret

LBL(.L__Y_near_zero):
	vmovss	.L4_101(%rip), %xmm0
	ret

/* -------------------------------------------------------------------------- */

        ELF_FUNC(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA))


/* ============================================================
 *  fastexpf.s
 * 
 *  An implementation of the expf libm function.
 * 
 *  Prototype:
 * 
 *      float fastexpf(float x);
 * 
 *    Computes e raised to the x power.
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

	.text
        ALN_FUNC
	.globl ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

	vcomiss %xmm0, %xmm0
	jp	LBL(.LB_NZERO_SS_VEX)

        vcomiss .L__np_ln_lead_table(%rip), %xmm0        /* Equal to 0.0? */
        jne     LBL(.LB_NZERO_SS_VEX)
        vmovss .L4_386(%rip), %xmm0
        RZ_POP
        rep
        ret

LBL(.LB_NZERO_SS_VEX):

        vcomiss .L__sp_ln_max_singleval(%rip), %xmm0
        ja      LBL(.L_sp_inf)
        vcomiss .L_real_min_singleval(%rip), %xmm0
        jbe     LBL(.L_sp_ninf)


        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
        /* Step 1. Reduce the argument. */
        /* r = x * thirtytwo_by_logbaseof2; */

/*      vunpcklps %xmm0, %xmm0, %xmm0
        vcvtps2pd %xmm0, %xmm2
        vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm3 */
        vmovaps %xmm0, %xmm2
        vmovaps .L_s_real_thirtytwo_by_log2(%rip),%xmm3
/*      vmulsd  %xmm2,%xmm3,%xmm3 */
        vmulss  %xmm0,%xmm3,%xmm3

        /* Set n = nearest integer to r */
/*      vcvtpd2dq %xmm3,%xmm4 */        /* convert to integer */
/*      vcvtdq2pd %xmm4,%xmm1 */        /* and back to float. */

        vcvtps2dq %xmm3,%xmm4   /* convert to integer */
        vcvtdq2ps %xmm4,%xmm1   /* and back to float. */

        /* r1 = x - n * logbaseof2_by_32_lead; */
/* #ifdef TARGET_FMA
#        VFNMADDSD       %xmm2,.L__real_log2_by_32(%rip),%xmm1,%xmm2
	VFNMA_231SD	(.L__real_log2_by_32(%rip),%xmm1,%xmm2)
#else
        vmulsd  .L__real_log2_by_32(%rip),%xmm1,%xmm1 */
/*      vsubsd  %xmm1,%xmm2,%xmm2 */    /* r1 in xmm2, */
/* #endif */

#ifdef TARGET_FMA
#        VFNMADDSS       %xmm2,.L_s_real_log2_by_32(%rip),%xmm1,%xmm2
	VFNMA_231SS	(.L_s_real_log2_by_32(%rip),%xmm1,%xmm2)
#else
        vmulss  .L_s_real_log2_by_32(%rip),%xmm1,%xmm1
        vsubss  %xmm1,%xmm2,%xmm2       /* r1 in xmm2, */
#endif
        vmovd   %xmm4,%ecx
/*      leaq    .L__two_to_jby32_table(%rip),%rdx */
        leaq    .L_s_two_to_jby32_table(%rip),%rdx

        /* j = n & 0x0000001f; */
        movq    $0x1f,%rax
        and     %ecx,%eax

        /* f1 = .L__two_to_jby32_lead_table[j];  */
        /* f2 = .L__two_to_jby32_trail_table[j]; */
        /* *m = (n - j) / 32; */
        sub     %eax,%ecx
        sar     $5,%ecx

        /* Step 2. Compute the polynomial. */
        /* q = r1 + (r2 +
           r*r*( 5.00000000000000008883e-01 +
           r*( 1.66666666665260878863e-01 +
           r*( 4.16666666662260795726e-02 +
           r*( 8.33336798434219616221e-03 +
           r*( 1.38889490863777199667e-03 ))))));
           q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
/*      vmovsd  .L__real_3FC5555555548F7C(%rip),%xmm1   */      /* 1/6 */
/*      vmovapd %xmm2,%xmm0 */
        vmovss  .L_s_real_3FC5555555548F7C(%rip),%xmm1          /* 1/6 */
        vmovaps %xmm2,%xmm0

/* #ifdef TARGET_FMA
#        VFMADDSD        .L__real_3fe0000000000000(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm1)
        vmulsd          %xmm2,%xmm2,%xmm2
#        VFMADDSD        %xmm0,%xmm1,%xmm2,%xmm2
	VFMA_213SD	(%xmm0,%xmm1,%xmm2)
#else */
/*      vmulsd  %xmm2,%xmm1,%xmm1       */                      /* r/6 */
/*      vmulsd  %xmm2,%xmm2,%xmm2       */                      /* r^2 */
/*      vaddsd  .L__real_3fe0000000000000(%rip),%xmm1,%xmm1 */  /* 1/2+r/6 */
/*      vmulsd  %xmm1,%xmm2,%xmm2       */                      /* r^2/2+r^3/6 */
/*      vaddsd  %xmm0,%xmm2,%xmm2       */                      /* q=r+r^2/2+r^3/6 */
/* #endif */

#ifdef TARGET_FMA
#        VFMADDSS        .L_s_real_3fe0000000000000(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SS	(.L_s_real_3fe0000000000000(%rip),%xmm2,%xmm1)
        vmulss          %xmm2,%xmm2,%xmm2
#        VFMADDSS        %xmm0,%xmm1,%xmm2,%xmm2
	VFMA_213SS	(%xmm0,%xmm1,%xmm2)
#else
        vmulss  %xmm2,%xmm1,%xmm1                               /* r/6 */
        vmulss  %xmm2,%xmm2,%xmm2                               /* r^2 */
        vaddss  .L_s_real_3fe0000000000000(%rip),%xmm1,%xmm1    /* 1/2+r/6 */
        vmulss  %xmm1,%xmm2,%xmm2                               /* r^2/2+r^3/6 */
        vaddss  %xmm0,%xmm2,%xmm2                               /* q=r+r^2/2+r^3/6 */
#endif

/*      vmovsd  (%rdx,%rax,8),%xmm4     */                      /* f1+f2 */
        vmovss  (%rdx,%rax,4),%xmm4                             /* f1+f2=2^(j/32) */

        /* *z2 = f2 + ((f1 + f2) * q); */
/*        add   $1023, %ecx */  /* add bias */

        /* added by wangz */
        mov     $1, %edx
        mov     $1, %eax
        add     $127, %ecx      /* add bias */
        cmovle  %ecx, %edx
        cmovle  %eax, %ecx

        /* deal with infinite results */
        /* deal with denormal results */
/*        shlq  $52,%rcx */       /* build 2^n */
        shl     $23, %ecx        /* build 2^n */
        add     $127, %edx
        shl     $23, %edx

/* #ifdef TARGET_FMA
#        VFMADDSD        %xmm4,%xmm2,%xmm4,%xmm2
	VFMA_213SD	(%xmm4,%xmm4,%xmm2)
#else */
/*      vmulsd  %xmm4,%xmm2,%xmm2 */                            /* (f1+f2)*q */
/*      vaddsd  %xmm4,%xmm2,%xmm2 */  /* z = z1 + z2   done with 1,2,3,4,5 */
/* #endif */

#ifdef TARGET_FMA
#        VFMADDSS        %xmm4,%xmm2,%xmm4,%xmm2
	VFMA_213SS	(%xmm4,%xmm4,%xmm2)
#else
        vmulss  %xmm4,%xmm2,%xmm2                               /* (f1+f2)*q */
        vaddss  %xmm4,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 --> (f1+f2)*(1+q) */
#endif

        /* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
        /* Step 3. Reconstitute. */
        mov     %edx, RZ_OFF(24)(%rsp)
        vmulss  RZ_OFF(24)(%rsp),%xmm2,%xmm0

/*      movq    %rcx,RZ_OFF(24)(%rsp) */        /* get 2^n to memory */
        mov     %ecx,RZ_OFF(24)(%rsp)   /* get 2^n to memory */
/*      vmulsd  RZ_OFF(24)(%rsp),%xmm2,%xmm2 */ /* result *= 2^n */
        vmulss  RZ_OFF(24)(%rsp),%xmm2,%xmm0    /* result *= 2^n */
/*      vunpcklpd %xmm2, %xmm2, %xmm2
        vcvtpd2ps %xmm2, %xmm0 */



LBL(.L_sp_final_check):
        RZ_POP
        rep
        ret

LBL(.L_sp_inf):
        vmovlps .L_sp_real_infinity(%rip),%xmm0,%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_ninf):
        jp      LBL(.L_sp_cvt_nan)
        xor     %eax, %eax
        vmovd   %eax,%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_sinh_ninf):
        jp      LBL(.L_sp_cvt_nan)
        vmovlps  .L_sp_real_ninfinity(%rip),%xmm0,%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_cosh_ninf):
        jp      LBL(.L_sp_cvt_nan)
        vmovlps  .L_sp_real_infinity(%rip),%xmm0,%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_cvt_nan):
        xor     %eax, %eax
        vmovd   %eax,%xmm1
/*      vmovsd  .L_real_cvt_nan(%rip),%xmm1 */
        vmovss  .L_real_cvt_nan(%rip),%xmm1
        vorps   %xmm1, %xmm0, %xmm0
        jmp     LBL(.L_sp_final_check)

        ELF_FUNC(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA))




/* ============================================================
 *  vector fastexpf.s
 * 
 *  An implementation of the expf libm function.
 * 
 *  Prototype:
 * 
 *      float fastexpf(float x);
 * 
 *    Computes e raised to the x power.
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

	.text
        ALN_FUNC
	.globl ENT(ASM_CONCAT(__rvs_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rvs_exp_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64) || defined(TARGET_INTERIX_X8664)
	vmovdqu	%ymm6, RZ_OFF(104)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
#endif



        /* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
        /* Step 1. Reduce the argument. */
        /* r = x * thirtytwo_by_logbaseof2; */
/*      vmovhlps  %xmm0, %xmm1, %xmm1 */
        vmovaps  %xmm0, %xmm5
/*      vcvtps2pd %xmm0, %xmm2  */              /* xmm2 = dble(a(2)), dble(a(1)) */
/*      vcvtps2pd %xmm1, %xmm1  */              /* xmm1 = dble(a(4)), dble(a(3)) */
        vandps   .L__ps_mask_unsign(%rip), %xmm5, %xmm5
        vmovaps .L_s_real_thirtytwo_by_log2(%rip),%xmm3
/*      vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm3 */
/*      vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm4 */
        vcmpps  $6, .L__sp_ln_max_singleval(%rip), %xmm5, %xmm5
        vmulps  %xmm0, %xmm3, %xmm3
/*      vmulpd  %xmm2, %xmm3, %xmm3
        vmulpd  %xmm1, %xmm4, %xmm4 */
        vmovmskps %xmm5, %r8d

        /* Set n = nearest integer to r */
/*      vcvtpd2dq %xmm3,%xmm5   */      /* convert to integer */
/*      vcvtpd2dq %xmm4,%xmm6   */      /* convert to integer */
        vcvtps2dq %xmm3, %xmm5  /* convert to integer */
        test     $15, %r8d
        vcvtdq2ps %xmm5,%xmm3   /* and back to float. */
/*      vcvtdq2pd %xmm5,%xmm3   */      /* and back to float. */
/*      vcvtdq2pd %xmm6,%xmm4   */      /* and back to float. */
        jnz     LBL(.L__Scalar_fvsexp)

        /* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#        VFNMADDPS       %xmm0,.L_s_real_log2_by_32(%rip),%xmm3,%xmm0
	VFNMA_231PS	(.L_s_real_log2_by_32(%rip),%xmm3,%xmm0)
/*      VFNMADDPD       %xmm2,.L__real_log2_by_32(%rip),%xmm3,%xmm2 */
/*      VFNMADDPD       %xmm1,.L__real_log2_by_32(%rip),%xmm4,%xmm1 */
#else
        vmulps  .L_s_real_log2_by_32(%rip),%xmm3,%xmm3
        vsubps  %xmm3,%xmm0,%xmm0       /* r1 in xmm2, */

/*      vmulpd  .L__real_log2_by_32(%rip),%xmm3,%xmm3 */
/*      vsubpd  %xmm3,%xmm2,%xmm2 */    /* r1 in xmm2, */
/*      vmulpd  .L__real_log2_by_32(%rip),%xmm4,%xmm4 */
/*      vsubpd  %xmm4,%xmm1,%xmm1 */    /* r1 in xmm1, */
#endif
        vmovups %xmm5,RZ_OFF(24)(%rsp)
/*      vmovq   %xmm5,RZ_OFF(16)(%rsp) */
/*      vmovq   %xmm6,RZ_OFF(24)(%rsp) */

/*      vsubpd  %xmm3,%xmm2,%xmm2 */    /* r1 in xmm2, */
/*      vsubpd  %xmm4,%xmm1,%xmm1 */    /* r1 in xmm1, */
        leaq    .L_s_two_to_jby32_table(%rip),%rax

        /* j = n & 0x0000001f; */
        mov     RZ_OFF(12)(%rsp),%r8d
        mov     RZ_OFF(16)(%rsp),%r9d
        mov     RZ_OFF(20)(%rsp),%r10d
        mov     RZ_OFF(24)(%rsp),%r11d
/*      mov     RZ_OFF(20)(%rsp),%r10d
        mov     RZ_OFF(24)(%rsp),%r11d */

        movq    $0x1f, %rcx
        and     %r8d, %ecx
        movq    $0x1f, %rdx
        and     %r9d, %edx
        vmovaps %xmm0,%xmm2
        vmovaps %xmm0,%xmm4

/*      vmovapd %xmm2,%xmm0
        vmovapd %xmm1,%xmm3
        vmovapd %xmm2,%xmm4
        vmovapd %xmm1,%xmm5 */

        movq    $0x1f, %rsi
        and     %r10d, %esi
        movq    $0x1f, %rdi
        and     %r11d, %edi

        sub     %ecx,%r8d
        sar     $5,%r8d
        sub     %edx,%r9d
        sar     $5,%r9d

        /* Step 2. Compute the polynomial. */
        /* q = r1 + (r2 +
           r*r*( 5.00000000000000008883e-01 +
           r*( 1.66666666665260878863e-01 +
           r*( 4.16666666662260795726e-02 +
           r*( 8.33336798434219616221e-03 +
           r*( 1.38889490863777199667e-03 ))))));
           q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
        vmulps  .L_s_real_3FC5555555548F7C(%rip),%xmm0,%xmm0
/*      vmulpd  .L__real_3FC5555555548F7C(%rip),%xmm0,%xmm0
        vmulpd  .L__real_3FC5555555548F7C(%rip),%xmm1,%xmm1 */

        sub     %esi,%r10d
        sar     $5,%r10d
        sub     %edi,%r11d
        sar     $5,%r11d

        vmulps  %xmm2,%xmm2,%xmm2
/*      vmulpd  %xmm2,%xmm2,%xmm2
        vmulpd  %xmm3,%xmm3,%xmm3 */
        vaddps  .L_s_real_3fe0000000000000(%rip),%xmm0,%xmm0
/*      vaddpd  .L__real_3fe0000000000000(%rip),%xmm0,%xmm0
        vaddpd  .L__real_3fe0000000000000(%rip),%xmm1,%xmm1 */
#ifdef TARGET_FMA
#        VFMADDPS        %xmm4,%xmm0,%xmm2,%xmm2
	VFMA_213PS	(%xmm4,%xmm0,%xmm2)
/*      VFMADDPD        %xmm4,%xmm0,%xmm2,%xmm2 */
/*      VFMADDPD        %xmm5,%xmm1,%xmm3,%xmm3 */
#else
        vmulps  %xmm0,%xmm2,%xmm2
        vaddps  %xmm4,%xmm2,%xmm2
/*      vmulpd  %xmm0,%xmm2,%xmm2
        vaddpd  %xmm4,%xmm2,%xmm2
        vmulpd  %xmm1,%xmm3,%xmm3
        vaddpd  %xmm5,%xmm3,%xmm3 */
#endif
        vmovss  (%rax,%rdx,4),%xmm0
        vmovhps (%rax,%rcx,4),%xmm0,%xmm0
        vmovss  (%rax,%rdi,4),%xmm1
        vmovhps (%rax,%rsi,4),%xmm1,%xmm1
        vshufps $136, %xmm0, %xmm1, %xmm0

/*      vmovsd  (%rax,%rdx,8),%xmm0
        vmovhpd (%rax,%rcx,8),%xmm0,%xmm0

        vmovsd  (%rax,%rdi,8),%xmm1
        vmovhpd (%rax,%rsi,8),%xmm1,%xmm1 */

/*      vaddpd  %xmm4,%xmm2,%xmm2 */
/*      vaddpd  %xmm5,%xmm3,%xmm3 */

        /* *z2 = f2 + ((f1 + f2) * q); */
/*      vmulpd  %xmm0,%xmm2,%xmm2 */
/*      vmulpd  %xmm1,%xmm3,%xmm3 */

        /* deal with infinite and denormal results */
/*        add   $1023, %r8d     */      /* add bias */
/*        add   $1023, %r9d     */      /* add bias */
/*        add   $1023, %r10d    */      /* add bias */
/*        add   $1023, %r11d    */      /* add bias */

	mov	$1, %ecx
	mov	$1, %edx
	mov	$1, %esi
	mov	$1, %edi

	mov	$1, %eax

        add     $127, %r8d      /* add bias */
	cmovle	%r8d, %ecx
	cmovle	%eax, %r8d
        add     $127, %r9d      /* add bias */
	cmovle	%r9d, %edx
	cmovle	%eax, %r9d
        add     $127, %r10d     /* add bias */
	cmovle	%r10d, %esi
	cmovle	%eax, %r10d
        add     $127, %r11d     /* add bias */
	cmovle	%r11d, %edi
	cmovle	%eax, %r11d

/*      shlq    $52,%r8
        shlq    $52,%r9
        shlq    $52,%r10
        shlq    $52,%r11 */

        shl     $23,%r8d
        shl     $23,%r9d
        shl     $23,%r10d
        shl     $23,%r11d

        add     $127, %ecx      /* add bias */
        add     $127, %edx      /* add bias */
        add     $127, %esi     /* add bias */
        add     $127, %edi     /* add bias */

	shl	$23, %ecx
	shl	$23, %edx
	shl	$23, %esi
	shl	$23, %edi

#ifdef TARGET_FMA
#        VFMADDPS        %xmm0,%xmm0,%xmm2,%xmm2
	VFMA_213PS	(%xmm0,%xmm0,%xmm2)
/*      VFMADDPD        %xmm0,%xmm0,%xmm2,%xmm2 */
/*      VFMADDPD        %xmm1,%xmm1,%xmm3,%xmm3 */
#else
        vmulps  %xmm0,%xmm2,%xmm2
        vaddps  %xmm0,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
/*      vmulpd  %xmm0,%xmm2,%xmm2
        vaddpd  %xmm0,%xmm2,%xmm2 */  /* z = z1 + z2   done with 1,2,3,4,5 */
/*      vmulpd  %xmm1,%xmm3,%xmm3
        vaddpd  %xmm1,%xmm3,%xmm3 */  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

        /* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
        /* Step 3. Reconstitute. */
	mov	%edx,RZ_OFF(32)(%rsp)   /* get 2^n to memory */
        mov     %ecx,RZ_OFF(28)(%rsp)   /* get 2^n to memory */
        mov     %edi,RZ_OFF(40)(%rsp)  /* get 2^n to memory */
        mov     %esi,RZ_OFF(36)(%rsp)  /* get 2^n to memory */
        vmulps  RZ_OFF(40)(%rsp),%xmm2,%xmm0    /* result *= 2^n */

/*      movq    %r9,RZ_OFF(24)(%rsp) */   /* get 2^n to memory */
/*      movq    %r8,RZ_OFF(16)(%rsp) */    /* get 2^n to memory */
/*      vmulpd  RZ_OFF(24)(%rsp),%xmm2,%xmm2 */ /* result *= 2^n */
        mov     %r9d,RZ_OFF(32)(%rsp)   /* get 2^n to memory */
        mov     %r8d,RZ_OFF(28)(%rsp)   /* get 2^n to memory */

/*      movq    %r11,RZ_OFF(40)(%rsp) */  /* get 2^n to memory */
/*      movq    %r10,RZ_OFF(32)(%rsp) */  /* get 2^n to memory */
/*      vmulpd  RZ_OFF(40)(%rsp),%xmm3,%xmm3 */ /* result *= 2^n */
        mov     %r11d,RZ_OFF(40)(%rsp)  /* get 2^n to memory */
        mov     %r10d,RZ_OFF(36)(%rsp)  /* get 2^n to memory */
        vmulps  RZ_OFF(40)(%rsp),%xmm2,%xmm0    /* result *= 2^n */

/*      vcvtpd2ps %xmm2,%xmm0
        vcvtpd2ps %xmm3,%xmm1
        vshufps $68,%xmm1,%xmm0,%xmm0 */


LBL(.L_vsp_final_check):

#if defined(_WIN64) || defined(TARGET_INTERIX_X8664)
	vmovdqu	RZ_OFF(104)(%rsp), %ymm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif

	RZ_POP
	rep
	ret

LBL(.L__Scalar_fvsexp):
        pushq   %rbp			/* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovaps  %xmm0, _SX0(%rsp)

        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovss   _SX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovss   _SX2(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovss   _SX3(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

LBL(.L__final_check):
        RZ_POP
        rep
        ret

        ELF_FUNC(ASM_CONCAT(__rvs_exp_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rvs_exp_,TARGET_VEX_OR_FMA))



/* ------------------------------------------------------------------------- */
/* 
 *  vector sinle precision exp
 * 
 *  Prototype:
 * 
 *      single __rvs_exp_vex/fma4_256(float *x);
 * 
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__rvs_exp_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__rvs_exp_,TARGET_VEX_OR_FMA,_256)):

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

	movq	%r12, 216(%rsp)
	movq	%r13, 224(%rsp)
	movq	%r14, 232(%rsp)
	movq	%r15, 240(%rsp)

#if defined(_WIN64) || defined(TARGET_INTERIX_X8664)
        vmovdqu %ymm6, 128(%rsp)
        movq    %rsi, 200(%rsp)
        movq    %rdi, 208(%rsp)
#endif

	vmovaps  %ymm0, %ymm5
	vandps   .L__ps_mask_unsign(%rip), %ymm5, %ymm5
	vmovups .L_s_real_thirtytwo_by_log2(%rip),%ymm3
	vcmpps  $6, .L__sp_ln_max_singleval(%rip), %ymm5, %ymm5
	vmulps  %ymm0, %ymm3, %ymm3
	vmovmskps %ymm5, %r8d

	vcvtps2dq %ymm3, %ymm5
	test     $255, %r8d
	vcvtdq2ps %ymm5,%ymm3

	jnz     LBL(.L__Scalar_fvsexp_256)

        /* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#        VFNMADDPS       %ymm0,.L_s_real_log2_by_32(%rip),%ymm3,%ymm0
	VFNMA_231PS	(.L_s_real_log2_by_32(%rip),%ymm3,%ymm0)
#else
        vmulps  .L_s_real_log2_by_32(%rip),%ymm3,%ymm3
        vsubps  %ymm3,%ymm0,%ymm0       /* r1 in xmm2, */
#endif

	vmovups %ymm5,RZ_OFF(40)(%rsp)
	leaq    .L_s_two_to_jby32_table(%rip),%rax

	/* j = n & 0x0000001f; */
        mov     RZ_OFF(12)(%rsp),%r8d
        mov     RZ_OFF(16)(%rsp),%r9d
        mov     RZ_OFF(20)(%rsp),%r10d
        mov     RZ_OFF(24)(%rsp),%r11d

        mov     RZ_OFF(28)(%rsp),%r12d
        mov     RZ_OFF(32)(%rsp),%r13d
        mov     RZ_OFF(36)(%rsp),%r14d
        mov     RZ_OFF(40)(%rsp),%r15d

        movq    $0x1f, %rcx
        and     %r8d, %ecx
        movq    $0x1f, %rdx
        and     %r9d, %edx

        vmovaps %ymm0,%ymm2
        vmovaps %ymm0,%ymm4

        movq    $0x1f, %rsi
        and     %r10d, %esi
        movq    $0x1f, %rdi
        and     %r11d, %edi

        sub     %ecx,%r8d
        sar     $5,%r8d
        sub     %edx,%r9d
        sar     $5,%r9d

        /* Step 2. Compute the polynomial. */
        /* q = r1 + (r2 +       
           r*r*( 5.00000000000000008883e-01 +
           r*( 1.66666666665260878863e-01 +
           r*( 4.16666666662260795726e-02 +
           r*( 8.33336798434219616221e-03 +
           r*( 1.38889490863777199667e-03 ))))));
           q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
        vmulps  .L_s_real_3FC5555555548F7C(%rip),%ymm0,%ymm0

        sub     %esi,%r10d
        sar     $5,%r10d
        sub     %edi,%r11d
        sar     $5,%r11d

        vmulps  %ymm2,%ymm2,%ymm2
        vaddps  .L_s_real_3fe0000000000000(%rip),%ymm0,%ymm0

#ifdef TARGET_FMA
#        VFMADDPS        %ymm4,%ymm0,%ymm2,%ymm2
	VFMA_213PS	(%ymm4,%ymm0,%ymm2)
#else
        vmulps  %ymm0,%ymm2,%ymm2
        vaddps  %ymm4,%ymm2,%ymm2
#endif
        vmovss  (%rax,%rdx,4),%xmm0
        vmovhps (%rax,%rcx,4),%xmm0,%xmm0
        vmovss  (%rax,%rdi,4),%xmm1
        vmovhps (%rax,%rsi,4),%xmm1,%xmm1

        vshufps $136, %xmm0, %xmm1, %xmm0

        movq    $0x1f, %rcx
        and     %r12d, %ecx
        movq    $0x1f, %rdx
        and     %r13d, %edx

        movq    $0x1f, %rsi
        and     %r14d, %esi
        movq    $0x1f, %rdi
        and     %r15d, %edi

        sub     %ecx,%r12d
        sar     $5,%r12d
        sub     %edx,%r13d
        sar     $5,%r13d

        sub     %esi,%r14d
        sar     $5,%r14d
        sub     %edi,%r15d
        sar     $5,%r15d

        vmovss  (%rax,%rdx,4),%xmm3
        vmovhps (%rax,%rcx,4),%xmm3,%xmm3
        vmovss  (%rax,%rdi,4),%xmm4
        vmovhps (%rax,%rsi,4),%xmm4,%xmm4

        vshufps $136, %xmm3, %xmm4, %xmm3

	vinsertf128	$1, %xmm0, %ymm3, %ymm0

#ifdef TARGET_FMA
#        VFMADDPS        %ymm0,%ymm0,%ymm2,%ymm2
	VFMA_213PS	(%ymm0,%ymm0,%ymm2)
#else
        vmulps  %ymm0,%ymm2,%ymm2
        vaddps  %ymm0,%ymm2,%ymm2  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

	mov	$1, %ecx
	mov	$1, %edx
	mov	$1, %esi
	mov	$1, %edi

	mov	$1, %eax

	add	$127, %r8d
	cmovle	%r8d, %ecx
        cmovle  %eax, %r8d
        add     $127, %r9d      /* add bias */
        cmovle  %r9d, %edx
        cmovle  %eax, %r9d
        add     $127, %r10d     /* add bias */
        cmovle  %r10d, %esi
        cmovle  %eax, %r10d
        add     $127, %r11d     /* add bias */
        cmovle  %r11d, %edi
        cmovle  %eax, %r11d	

        shl     $23,%r8d
        shl     $23,%r9d
        shl     $23,%r10d
        shl     $23,%r11d

	add     $127, %ecx      /* add bias */
        add     $127, %edx      /* add bias */
        add     $127, %esi     /* add bias */
        add     $127, %edi     /* add bias */

        shl     $23, %ecx
        shl     $23, %edx
        shl     $23, %esi
        shl     $23, %edi

        mov     %r8d,RZ_OFF(44)(%rsp)   /* get 2^n to memory */
        mov     %r9d,RZ_OFF(48)(%rsp)   /* get 2^n to memory */
        mov     %r10d,RZ_OFF(52)(%rsp)  /* get 2^n to memory */
        mov     %r11d,RZ_OFF(56)(%rsp)  /* get 2^n to memory */

	mov	%edx,RZ_OFF(4)(%rsp)
	mov	%ecx,RZ_OFF(8)(%rsp)
	mov	%edi,RZ_OFF(12)(%rsp)
	mov	%esi,RZ_OFF(16)(%rsp)

        mov     $1, %ecx
        mov     $1, %edx
        mov     $1, %esi
        mov     $1, %edi

        add     $127, %r12d
        cmovle  %r12d, %ecx
        cmovle  %eax, %r12d
        add     $127, %r13d      /* add bias */
        cmovle  %r13d, %edx
        cmovle  %eax, %r13d
        add     $127, %r14d     /* add bias */
        cmovle  %r14d, %esi
        cmovle  %eax, %r14d
        add     $127, %r15d     /* add bias */
        cmovle  %r15d, %edi
        cmovle  %eax, %r15d

        shl     $23,%r12d
        shl     $23,%r13d
        shl     $23,%r14d
        shl     $23,%r15d

        add     $127, %ecx      /* add bias */
        add     $127, %edx      /* add bias */
        add     $127, %esi     /* add bias */
        add     $127, %edi     /* add bias */

        shl     $23, %ecx
        shl     $23, %edx
        shl     $23, %esi
        shl     $23, %edi

	mov     %edx,RZ_OFF(20)(%rsp)
        mov     %ecx,RZ_OFF(24)(%rsp)
        mov     %edi,RZ_OFF(28)(%rsp)
        mov     %esi,RZ_OFF(32)(%rsp)
        vmulps  RZ_OFF(32)(%rsp),%ymm2,%ymm0    /* result *= 2^n */

        mov     %r12d,RZ_OFF(60)(%rsp)   /* get 2^n to memory */
        mov     %r13d,RZ_OFF(64)(%rsp)   /* get 2^n to memory */
        mov     %r14d,RZ_OFF(68)(%rsp)  /* get 2^n to memory */
        mov     %r15d,RZ_OFF(72)(%rsp)  /* get 2^n to memory */
        vmulps  RZ_OFF(72)(%rsp),%ymm2,%ymm0    /* result *= 2^n */

LBL(.L_vsp_final_check_256):

#if defined(_WIN64) || defined(TARGET_INTERIX_X8664)
        vmovdqu 128(%rsp), %ymm6
        movq    200(%rsp), %rsi
        movq    208(%rsp), %rdi
#endif
	movq	216(%rsp), %r12
	movq	224(%rsp), %r13
	movq	232(%rsp), %r14
	movq	240(%rsp), %r15

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsexp_256):
        pushq   %rbp                    /* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $256, %rsp
        vmovups  %ymm0, 0(%rsp)

        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 32(%rsp)

        vmovss   4(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 36(%rsp)

        vmovss   8(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 40(%rsp)

        vmovss   12(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 44(%rsp)

        vmovss   16(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 48(%rsp)

        vmovss   20(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 52(%rsp)

        vmovss   24(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 56(%rsp)

        vmovss   28(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__rss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 60(%rsp)

        vmovups  32(%rsp), %ymm0
        movq    %rbp, %rsp
        popq    %rbp

	movq	216(%rsp), %r12
	movq	224(%rsp), %r13
	movq	232(%rsp), %r14
	movq	240(%rsp), %r15

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__rvs_exp_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__rvs_exp_,TARGET_VEX_OR_FMA,_256))






/* ------------------------------------------------------------------------- */
/* 
 *  vector sinle precision pow
 * 
 *  Prototype:
 * 
 *      single __rvs_pow_vex/fma4_256(float *x);
 * 
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__rvs_pow_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__rvs_pow_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

/********************************************/

        vmovaps %ymm0, %ymm4
        vmovaps %ymm1, %ymm5
        vmovaps  %ymm0, %ymm2
        vxorps  %ymm3, %ymm3, %ymm3
        vandps  .L4_fvspow_infinity_mask(%rip), %ymm4, %ymm4
        vandps  .L4_fvspow_infinity_mask(%rip), %ymm5, %ymm5
        vcmpps  $2, %ymm3, %ymm2, %ymm2
        vcmpps  $0, .L4_fvspow_infinity_mask(%rip), %ymm4, %ymm4
        vcmpps  $0, .L4_fvspow_infinity_mask(%rip), %ymm5, %ymm5
        vorps   %ymm4, %ymm2, %ymm2
        /* Store input arguments onto stack */
        vmovups  %ymm0, 0(%rsp)
        vorps   %ymm5, %ymm2, %ymm2
        vmovups  %ymm1, 32(%rsp)
        vmovmskps %ymm2, %r8d
        test    $255, %r8d
        jnz     LBL(.L__Scalar_fvspow_256)

        CALL(ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256)))


        vmovups  32(%rsp), %ymm1
        vmulps  %ymm1, %ymm0, %ymm0

        CALL(ENT(ASM_CONCAT3(__rvs_exp_,TARGET_VEX_OR_FMA,_256)))


        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvspow_256):
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 64(%rsp)

        vmovss   4(%rsp), %xmm0
        vmovss   36(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 68(%rsp)

        vmovss   8(%rsp), %xmm0
        vmovss   40(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 72(%rsp)

        vmovss   12(%rsp), %xmm0
        vmovss   44(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 76(%rsp)

        vmovss   16(%rsp), %xmm0
        vmovss   48(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 80(%rsp)

        vmovss   20(%rsp), %xmm0
        vmovss   52(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 84(%rsp)

        vmovss   24(%rsp), %xmm0
        vmovss   56(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 88(%rsp)

        vmovss   28(%rsp), %xmm0
        vmovss   60(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__rss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, 92(%rsp)

        vmovups  64(%rsp), %ymm0
        movq    %rbp, %rsp
        popq    %rbp
        ret


/********************************************/

        ELF_FUNC(ASM_CONCAT3(__rvs_pow_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__rvs_pow_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/* 
 *  vector double precision exp(relaxed)
 * 
 *  Prototype:
 * 
 *      double __rvd_exp_vex/fma4(double *x);
 * 
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__rvd_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rvd_exp_,TARGET_VEX_OR_FMA)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

        vmovups %xmm0, 8(%rsp)

        CALL(ENT(ASM_CONCAT(__rsd_exp_,TARGET_VEX_OR_FMA)))

        vmovsd %xmm0, 24(%rsp)
        vmovsd 16(%rsp), %xmm0

        CALL(ENT(ASM_CONCAT(__rsd_exp_,TARGET_VEX_OR_FMA)))


        vmovsd %xmm0, 32(%rsp)
        vmovups 24(%rsp), %xmm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT(__rvd_exp_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rvd_exp_,TARGET_VEX_OR_FMA))




/* ------------------------------------------------------------------------- */
/* 
 *  vector double precision exp(relaxed)
 * 
 *  Prototype:
 * 
 *      double __rvd_exp_vex/fma4_256(double *x);
 * 
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__rvd_exp_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__rvd_exp_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $80, %rsp

        vmovups %ymm0, 8(%rsp)

        CALL(ENT(ASM_CONCAT(__rvd_exp_,TARGET_VEX_OR_FMA)))


        vmovups %xmm0, 40(%rsp)
        vmovups 8(%rsp), %ymm2
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0

        CALL(ENT(ASM_CONCAT(__rvd_exp_,TARGET_VEX_OR_FMA)))

        vmovups 40(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__rvd_exp_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__rvd_exp_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */

/* 
 *  vector single precision tangent - 128
 * 
 *  Prototype:
 * 
 *      single __rvs_tan_vex/fma4(float *x);
 * 
 */

/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__rvs_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rvs_tan_,TARGET_VEX_OR_FMA)):


        subq $8, %rsp

        CALL(ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)))


        vdivps  %xmm1, %xmm0, %xmm0

        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__rvs_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rvs_tan_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */

/* 
 *  vector single precision tangent - 256
 * 
 *  Prototype:
 * 
 *      single __rvs_tan_vex/fma4_256(float *x);
 * 
 */

/* ------------------------------------------------------------------------- */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__rvs_tan_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__rvs_tan_,TARGET_VEX_OR_FMA,_256)):


        subq    $136, %rsp

        vmovups %ymm0, 32(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)))


        vmovups 32(%rsp), %ymm2
        vmovaps %xmm0, %xmm3
	vmovaps	%xmm1, %xmm4
        vextractf128    $1, %ymm2, %xmm0
        vmovups %xmm3, 64(%rsp)
	vmovups	%xmm4, 96(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)))

        vmovups 64(%rsp), %xmm3
        vinsertf128     $1, %xmm0, %ymm3, %ymm0
        vmovups 96(%rsp), %xmm4
        vinsertf128     $1, %xmm1, %ymm4, %ymm1

        vdivps %ymm1, %ymm0, %ymm0

        addq    $136, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__rvs_tan_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__rvs_tan_,TARGET_VEX_OR_FMA,_256))


/* ------------------------------------------------------------------------- */

/* 
 *  scalar single precision tangent
 * 
 *  Prototype:
 * 
 *      single __rss_tan_vex/fma4(float *x);
 * 
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__rss_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rss_tan_,TARGET_VEX_OR_FMA)):


        subq $8, %rsp

        CALL(ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA)))


        vdivss  %xmm1, %xmm0, %xmm0

        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__rss_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rss_tan_,TARGET_VEX_OR_FMA))



/* ------------------------------------------------------------------------- */

/* 
 *  vector double precision tangent
 * 
 *  Prototype:
 * 
 *      single __rvd_tan_vex/fma4(double *x);
 * 
 */

/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__rvd_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rvd_tan_,TARGET_VEX_OR_FMA)):


        subq $8, %rsp

        CALL(ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)))


        vdivpd  %xmm1, %xmm0, %xmm0

        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__rvd_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rvd_tan_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */


/* 
 *  vector double precision tangent
 * 
 *  Prototype:
 * 
 *      single __rvd_tan_vex/fma4_256(double *x);
 * 
 */

/* ------------------------------------------------------------------------- */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__rvd_tan_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__rvd_tan_,TARGET_VEX_OR_FMA,_256)):


        subq    $136, %rsp

        vmovupd %ymm0, 32(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)))


        vmovupd 32(%rsp), %ymm2
        vmovapd %xmm0, %xmm3
	vmovapd	%xmm1, %xmm4
        vextractf128    $1, %ymm2, %xmm0
        vmovupd %xmm3, 64(%rsp)
	vmovupd	%xmm4, 96(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)))

        vmovupd 64(%rsp), %xmm3
        vinsertf128     $1, %xmm0, %ymm3, %ymm0
        vmovupd 96(%rsp), %xmm4
        vinsertf128     $1, %xmm1, %ymm4, %ymm1

        vdivpd %ymm1, %ymm0, %ymm0

        addq    $136, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__rvd_tan_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__rvd_tan_,TARGET_VEX_OR_FMA,_256))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__rsd_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__rsd_tan_,TARGET_VEX_OR_FMA)):


        subq $8, %rsp

        CALL(ENT(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA)))


        vdivsd  %xmm1, %xmm0, %xmm0

        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__rsd_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__rsd_tan_,TARGET_VEX_OR_FMA))



