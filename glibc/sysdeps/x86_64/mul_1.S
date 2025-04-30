/* AMD64 __mpn_mul_1 -- Multiply a limb vector with a limb and store
   the result in a second limb vector.
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

#include <sysdep.h>
#include "asm-syntax.h"

#define rp	%rdi
#define up	%rsi
#define n_param	%rdx
#define vl	%rcx

#define n	%r11

	.text
ENTRY (__mpn_mul_1)
	push	%rbx
	cfi_adjust_cfa_offset (8)
	cfi_rel_offset (%rbx, 0)
	xor	%r10, %r10
	mov	(up), %rax		/* read first u limb early */
	mov	n_param, %rbx		/* move away n from rdx, mul uses it */
	mul	vl
	mov	%rbx, %r11

	add	%r10, %rax
	adc	$0, %rdx

	and	$3, %ebx
	jz	L(b0)
	cmp	$2, %ebx
	jz	L(b2)
	jg	L(b3)

L(b1):	dec	n
	jne	L(gt1)
	mov	%rax, (rp)
	jmp	L(ret)
L(gt1):	lea	8(up,n,8), up
	lea	-8(rp,n,8), rp
	neg	n
	xor	%r10, %r10
	xor	%ebx, %ebx
	mov	%rax, %r9
	mov	(up,n,8), %rax
	mov	%rdx, %r8
	jmp	L(L1)

L(b0):	lea	(up,n,8), up
	lea	-16(rp,n,8), rp
	neg	n
	xor	%r10, %r10
	mov	%rax, %r8
	mov	%rdx, %rbx
	jmp	L(L0)

L(b3):	lea	-8(up,n,8), up
	lea	-24(rp,n,8), rp
	neg	n
	mov	%rax, %rbx
	mov	%rdx, %r10
	jmp	L(L3)

L(b2):	lea	-16(up,n,8), up
	lea	-32(rp,n,8), rp
	neg	n
	xor	%r8, %r8
	xor	%ebx, %ebx
	mov	%rax, %r10
	mov	24(up,n,8), %rax
	mov	%rdx, %r9
	jmp	L(L2)

	.p2align 4
L(top): mov	%r10, (rp,n,8)
	add	%rax, %r9
	mov	(up,n,8), %rax
	adc	%rdx, %r8
	mov	$0, %r10d
L(L1):	mul	vl
	mov	%r9, 8(rp,n,8)
	add	%rax, %r8
	adc	%rdx, %rbx
L(L0):	mov	8(up,n,8), %rax
	mul	vl
	mov	%r8, 16(rp,n,8)
	add	%rax, %rbx
	adc	%rdx, %r10
L(L3):	mov	16(up,n,8), %rax
	mul	vl
	mov	%rbx, 24(rp,n,8)
	mov	$0, %r8d                # zero
	mov	%r8, %rbx               # zero
	add	%rax, %r10
	mov	24(up,n,8), %rax
	mov	%r8, %r9                # zero
	adc	%rdx, %r9
L(L2):	mul	vl
	add	$4, n
	js	L(top)

	mov	%r10, (rp,n,8)
	add	%rax, %r9
	adc	%r8, %rdx
	mov	%r9, 8(rp,n,8)
	add	%r8, %rdx
L(ret):	mov	%rdx, %rax

	pop	%rbx
	cfi_adjust_cfa_offset (-8)
	cfi_restore (%rbx)
	ret
END (__mpn_mul_1)
