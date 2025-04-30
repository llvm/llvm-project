/* x86-64 __mpn_addmul_1 -- Multiply a limb vector with a limb and add
   the result to a second limb vector.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU MP Library.

   The GNU MP Library is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or (at your
   option) any later version.

   The GNU MP Library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
   License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the GNU MP Library; see the file COPYING.LIB.  If not,
   see <https://www.gnu.org/licenses/>.  */

#include "sysdep.h"
#include "asm-syntax.h"

#define rp	%rdi
#define up	%rsi
#define n	%rdx
#define v0	%rcx

#ifndef func
# define func __mpn_addmul_1
# define ADDSUB add
#endif

	.text
ENTRY (func)
	push	%rbx
	push	%rbp
	lea	(%rdx), %rbx
	neg	%rbx

	mov	(up), %rax
	mov	(rp), %r10

	lea	-16(rp,%rdx,8), rp
	lea	(up,%rdx,8), up
	mul	%rcx

	bt	$0, %ebx
	jc	L(odd)

	lea	(%rax), %r11
	mov	8(up,%rbx,8), %rax
	lea	(%rdx), %rbp
	mul	%rcx
	add	$2, %rbx
	jns	L(n2)

	lea	(%rax), %r8
	mov	(up,%rbx,8), %rax
	lea	(%rdx), %r9
	jmp	L(mid)

L(odd):	add	$1, %rbx
	jns	L(n1)

	lea	(%rax), %r8
	mov	(up,%rbx,8), %rax
	lea	(%rdx), %r9
	mul	%rcx
	lea	(%rax), %r11
	mov	8(up,%rbx,8), %rax
	lea	(%rdx), %rbp
	jmp	L(e)

	.p2align 4
L(top):	mul	%rcx
	ADDSUB	%r8, %r10
	lea	(%rax), %r8
	mov	(up,%rbx,8), %rax
	adc	%r9, %r11
	mov	%r10, -8(rp,%rbx,8)
	mov	(rp,%rbx,8), %r10
	lea	(%rdx), %r9
	adc	$0, %rbp
L(mid):	mul	%rcx
	ADDSUB	%r11, %r10
	lea	(%rax), %r11
	mov	8(up,%rbx,8), %rax
	adc	%rbp, %r8
	mov	%r10, (rp,%rbx,8)
	mov	8(rp,%rbx,8), %r10
	lea	(%rdx), %rbp
	adc	$0, %r9
L(e):	add	$2, %rbx
	js	L(top)

	mul	%rcx
	ADDSUB	%r8, %r10
	adc	%r9, %r11
	mov	%r10, -8(rp)
	adc	$0, %rbp
L(n2):	mov	(rp), %r10
	ADDSUB	%r11, %r10
	adc	%rbp, %rax
	mov	%r10, (rp)
	adc	$0, %rdx
L(n1):	mov	8(rp), %r10
	ADDSUB	%rax, %r10
	mov	%r10, 8(rp)
	mov	%ebx, %eax	/* zero rax */
	adc	%rdx, %rax
	pop	%rbp
	pop	%rbx
	ret
END (func)
