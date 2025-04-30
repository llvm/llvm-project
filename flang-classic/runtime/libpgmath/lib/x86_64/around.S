/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include "directives.h"

	.text
	ALN_FUNC
	.globl ENT(__mth_i_around)
ENT(__mth_i_around):
	RZ_PUSH
	movss	%xmm0, RZ_OFF(4)(%rsp)
	movl	RZ_OFF(4)(%rsp), %eax
	shrl	$23, %eax
	andl	$255, %eax
	cmpl	$149, %eax
	jg	.L3
	cvtss2si	%xmm0, %eax
	cvtsi2ss	%eax, %xmm0
.L3:
	RZ_POP
	ret
	ELF_FUNC(__mth_i_around)
	ELF_SIZE(__mth_i_around)
