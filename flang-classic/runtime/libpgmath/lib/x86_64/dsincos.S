/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include "directives.h"

#if defined(TARGET_WIN_X8664)
#define IN F1
#define SS I2
#define CC I3
#else
#define IN F1
#define SS I1
#define CC I2
#endif

/*
 *  <double, double> __mth_i_dsincos_avx2(double f)
 *  <double, double> __mth_i_dsincos(double f)
 *
 *	Entry:	(%xmm0) = f
 *
 *	Exit	(%xmm0) = sin(f)
 *		(%xmm1) = cos(f)
 */

	.text
	ALN_FUNC
	.globl	ENT(__mth_i_dsincos_avx2)
ENT(__mth_i_dsincos_avx2):		/* %xmm0 contains f */

	movq	1f(%rip),%rax	/* Address of mth_dsincos_avx2 */
	jmp	2f		/* Jump to common __mth_i_dsincos() logic */

	ELF_FUNC(__mth_i_dsincos_avx2)
	ELF_SIZE(__mth_i_dsincos_avx2)

	.data
	ALN_QUAD
1:	.quad	ENT(__mth_dsincos_avx2)

	.text
	.globl	ENT(__mth_i_dsincos)
ENT(__mth_i_dsincos):		/* %xmm0 contains f */

	movq	1f(%rip),%rax	/* Address of mth_dsincos */

2:				/* If coming from __mth_dsincos_avx2 */
	subq	$24,%rsp
	leaq	(%rsp),SS	/* addr where to store double precision sin */
	leaq	8(%rsp),CC	/* addr where to store double precision cos */
	call	*%rax		/* call __mth_i_dsincos_avx2 or __mth_i_dsincos */
	movsd	0(%rsp),%xmm0	/* sin */
	movsd	8(%rsp),%xmm1	/* cos */
	addq	$24,%rsp
	ret

	ELF_FUNC(__mth_i_dsincos)
	ELF_SIZE(__mth_i_dsincos)

	.data
	ALN_QUAD
1:	.quad	ENT(__mth_dsincos)
