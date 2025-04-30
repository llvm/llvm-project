/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef BIAS
# define BIAS 0x7fff0000
#endif

	.section	.data.rel.local,"aw",@progbits
	.align 8
.Ljmp:
	.quad	func + BIAS
	.text
	.type	func, @function
func:
	.cfi_startproc
#if defined __CET__ && (__CET__ & 1) != 0
	endbr64
#endif
	xorl	%edi, %edi
	jmp	exit@PLT
	.cfi_endproc
	.size	func, .-func
	.globl	foo
	.type	foo, @function
foo:
	.cfi_startproc
	.cfi_def_cfa_register 6
#if defined __CET__ && (__CET__ & 1) != 0
	endbr64
#endif
	movq	.Ljmp(%rip), %rax
	subq	$BIAS, %rax
	jmp	*%rax
	.cfi_endproc
	.size	foo, .-foo
