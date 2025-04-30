/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#if 0
#if defined(_WIN64)
typedef long long I64;
typedef unsigned long long UI64; 
#else
typedef long I64;
typedef unsigned long UI64; 
#endif

typedef union {
    double f;
    UI64   i;
} FPI;

#define EXPBIAS 1023
#define MANTBITS 52
#define GET_EXP(u) (I64)(((u) & 0x7ff0000000000000) >> MANTBITS)

double
__mth_i_dint (double xx)
{
    I64          xexp;
    UI64         ux, mask;
    double       x;
    FPI         fpi;

    x = xx;
    fpi.f = x;
    ux = fpi.i;
    xexp = GET_EXP(ux) - EXPBIAS;
    if (xexp < 0 ) {
	/* |x| < 0  =>  zero with the original sign */
	fpi.i = ( ux & 0x8000000000000000 );
    }
    else if (xexp < MANTBITS) {
	/* 1 <= |x| < 2^53:
	 *    just mask out the trailing bits of the mantiassa beyond the
	 *    range of the exponent; mask out the exponent field as well.
	 */
	mask = ((UI64)1 << ( MANTBITS - xexp )) - 1;
	fpi.i = ux & ~mask;
    }
    /* else illegal input, nan, inf, overflow, ...; just return it */ 

    return fpi.f;
}
#endif

#include "directives.h"
/* Use is_<PROC-FEATURE>() functions from libpgc */
#include "x86id.h"

	.text
	ALN_FUNC
	.globl	ENT(__mth_i_dint)
ENT(__mth_i_dint):
/*
 *	Assembly based on Clang's translation of:
 *		xexp = GET_EXP(ux) - EXPBIAS;
 *		mask = fpi.i;
 *		mask = xexp < MANTBITS ? (-1ULL) << (MANTBITS - xexp) : mask;
 *		mask = xexp < 0 ? 0x8000000000000000 : mask;
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
	vroundsd $3, %xmm0, %xmm0, %xmm0
	ret

/*
 *	SSE4.1 version.
 */
2:
	testl	$HW_SSE4, %eax
	jz	3f		// Maybe prior to SSE4.1
	roundsd	$3, %xmm0, %xmm0
	ret


/*
 *	SSE version.
 *	Entry:	(%eax) = X86IDFN(hw_features)
 *		(%xmm0) = xx (argument)
 */
3:
	testl	$HW_SSE, %eax
	jz	4f		// Possibly X86IDFN(hw_features) undefined
	movq	%xmm0, %rdx
	movq	%rdx, %rax
	shrq	$52, %rdx
	andl	$2047, %edx
	movl	$1075, %ecx
	subl	%edx, %ecx
	movq	$-1, %r8
	shlq	%cl, %r8
	cmpq	$1075, %rdx
	cmovaeq	%rax, %r8
	cmpq	$1023, %rdx
	movabsq	$-9223372036854775808, %rcx
	cmovaeq	%r8, %rcx
	andq	%rax, %rcx
	movq	%rcx, %xmm0
	retq

/*
 *	X86IDFN(init_hw_features) is most likely uninitialized.
 *	(%eax) = current value of X86IDFN(init_hw_features).
 */
4:
	subq	$8, %rsp
	movsd	%xmm0, (%rsp)	// Save %xmm0 - do not use vmovsd
	movl	%eax, I1W	// Input to X86IDFN(init_hw_feature)()
	CALL	(ENT(X86IDFN(init_hw_features)))
//	(%eax) = hw_features
	movsd	(%rsp), %xmm0	// restore %xmm0 - do not use vmovsd
	addq	$8, %rsp
	jmp	1b

	ELF_FUNC(__mth_i_dint)
	ELF_SIZE(__mth_i_dint)
