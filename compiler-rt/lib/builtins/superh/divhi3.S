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


FUNCTION_BEGIN __sh_divhi3
	
FUNCTION_END __sh_divhi3