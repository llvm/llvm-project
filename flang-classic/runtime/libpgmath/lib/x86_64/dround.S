/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include "directives.h"

	.text
	ALN_FUNC
	.globl ENT(__mth_i_dround)
ENT(__mth_i_dround):
	RZ_PUSH
	movsd	%xmm0, RZ_OFF(8)(%rsp)
	movq	RZ_OFF(8)(%rsp), %rax
	shrq	$52, %rax
	andq	$2047, %rax
	cmpq	$1074, %rax
	jg	.L3
	cvtsd2si	%xmm0, %rax
	cvtsi2sd	%rax, %xmm0
.L3:
	RZ_POP
	ret
	ELF_FUNC(__mth_i_dround)
	ELF_SIZE(__mth_i_dround)
