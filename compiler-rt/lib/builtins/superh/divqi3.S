//===---------------------- SuperH builtin routine ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

	.macro FUNCTION_BEGIN name
	.text
		.p2align 4
	.globl \name
	.type \name, @function
\name:
	.endm

	.macro FUNCTION_END name
	.size  \name, . - \name
	.endm


FUNCTION_BEGIN __sh_divqi3
	shll16	r5
	shll8	r5
	exts.b  r4, r4
	tst 	r5, r5
	bt 		.Ldbz
	mov 	r4, r3
	rotcl 	r3
	subc 	r1, r1
	mov 	#0, r3
	subc 	r3, r4
	div0s 	r5, r1
	.rept 8
	rotcl   r4        
	div1    r5,r1
	.endr
	rotcl   r4
	addc    r3, r4
	exts.b  r4, r0
	mov 	#0, r1
	rts
	nop
.Ldbz:
	mov 	#0, r0
	mov 	r0, r1
	rts
	nop
FUNCTION_END __sh_divqi3