/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#if 0
typedef union {
    float        f;
    unsigned int i;
} FPI;

#define EXPBIAS  127
#define MANTBITS 23
#define GET_EXP(u) (int)(((u) & 0x7fffffff) >> MANTBITS)

float
__mth_i_aint( float xx )
{
    unsigned int  ux, mask;
    int           xexp;
    float         x;
    FPI          fpi;

    x = xx;
    fpi.f = x;
    ux = fpi.i;

    xexp = GET_EXP(ux) - EXPBIAS;
    if ( xexp < 0 ) {
	/* |x| < 0 => zero with the original sign */
	fpi.i = ux & 0x80000000;
    }
    else if ( xexp < MANTBITS ) {
	/* 1 <= |x| < 2^24:
	 *    just mask out the trailing bits of the mantissa beyond the
	 *    range of the exponent; mask out the exponent field as well.
	 */
	mask = (1 << ( MANTBITS - xexp )) - 1;
	fpi.i = ux & ~mask;
    }
    /* else for illegal input, nan, inf, overflow, ...; just return it */

    return fpi.f;
}

#endif

#include "directives.h"
/* Use is_<PROC-FEATURE>() functions from libpgc */
#include "x86id.h"

	.text
	ALN_FUNC
	.globl ENT(__mth_i_aint)
ENT(__mth_i_aint):
/*
 *	Assembly based on Clang's translation of:
 *		xexp = GET_EXP(ux) - EXPBIAS;
 *		mask = fpi.i;
 *		mask = xexp < MANTBITS ? (-1ULL) << (MANTBITS - xexp) : mask;
 *		mask = xexp < 0 ? 0x80000000 : mask;
 *		fpi.i &= mask;
 *
 *	Volatile registers common to both System V and Windows ABI:
 *	%rax, %rcx, %rdx, %r8, %r9
 */

#if  defined(TARGET_WIN_X8664)
	movl	ENT(X86IDFN(hw_features))(%rip), %eax
#else
	movq	ENT(X86IDFN(hw_features))@GOTPCREL(%rip), %rax
	movl	(%rax), %eax
#endif

1:
	testl	$HW_AVX, %eax
	jz	2f		// Maybe SSE4.1?

/*
 *	AVX version.
 */
	vroundss $3, %xmm0, %xmm0, %xmm0
	ret

/*
 *	SSE4.1 version.
 */
2:
	testl	$HW_SSE4, %eax
	jz	3f		// Maybe prior to SSE4.1
	roundss	$3, %xmm0, %xmm0
	ret

/*
 *	SSE version.
 */
3:
	testl	$HW_SSE, %eax
	jz	4f		// Possibly X86IDFN(hw_features) undefined
	movd	%xmm0, %eax
	movl	%eax, %ecx
	shrl	$23, %ecx
	movzbl	%cl, %edx
	movl	$150, %ecx
	subl	%edx, %ecx
	movl	$-1, %r8d
	shll	%cl, %r8d
	cmpl	$150, %edx
	cmovael	%eax, %r8d
	cmpl	$127, %edx
	movl	$-2147483648, %ecx
	cmovael	%r8d, %ecx
	andl	%eax, %ecx
	movd	%ecx, %xmm0
	retq

/*
 *	X86IDFN(init_hw_features) is most likely uninitialized.
 *	(%eax) = current value of X86IDFN(init_hw_features).
 */
4:
	subq	$8, %rsp
	movss	%xmm0, (%rsp)	// Save %xmm0 - do not use vmovss
	movl	%eax, I1W	// Input to X86IDFN(init_hw_feature)()
	CALL	(ENT(X86IDFN(init_hw_features)))
//	(%eax) = hw_features
	movss	(%rsp), %xmm0	// restore %xmm0 - do not use vmovss
	addq	$8, %rsp
	jmp	1b

	ELF_FUNC(__mth_i_aint)
	ELF_SIZE(__mth_i_aint)
