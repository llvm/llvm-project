/* x86-64 __mpn_add_n -- Add two limb vectors of the same length > 0 and store
   sum in a third limb vector.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#define vp	%rdx
#define n	%rcx
#define cy	%r8

#ifndef func
# define func __mpn_add_n
# define ADCSBB adc
#endif

	.text
ENTRY (func)
	xor	%r8, %r8
	mov	(up), %r10
	mov	(vp), %r11

	lea	-8(up,n,8), up
	lea	-8(vp,n,8), vp
	lea	-16(rp,n,8), rp
	mov	%ecx, %eax
	neg	n
	and	$3, %eax
	je	L(b00)
	add	%rax, n		/* clear low rcx bits for jrcxz */
	cmp	$2, %eax
	jl	L(b01)
	je	L(b10)

L(b11):	shr	%r8		/* set cy */
	jmp	L(e11)

L(b00):	shr	%r8		/* set cy */
	mov	%r10, %r8
	mov	%r11, %r9
	lea	4(n), n
	jmp	L(e00)

L(b01):	shr	%r8		/* set cy */
	jmp	L(e01)

L(b10):	shr	%r8		/* set cy */
	mov	%r10, %r8
	mov	%r11, %r9
	jmp	L(e10)

L(end):	ADCSBB	%r11, %r10
	mov	%r10, 8(rp)
	mov	%ecx, %eax	/* clear eax, ecx contains 0 */
	adc	%eax, %eax
	ret

	.p2align 4
L(top):
	mov	-24(up,n,8), %r8
	mov	-24(vp,n,8), %r9
	ADCSBB	%r11, %r10
	mov	%r10, -24(rp,n,8)
L(e00):
	mov	-16(up,n,8), %r10
	mov	-16(vp,n,8), %r11
	ADCSBB	%r9, %r8
	mov	%r8, -16(rp,n,8)
L(e11):
	mov	-8(up,n,8), %r8
	mov	-8(vp,n,8), %r9
	ADCSBB	%r11, %r10
	mov	%r10, -8(rp,n,8)
L(e10):
	mov	(up,n,8), %r10
	mov	(vp,n,8), %r11
	ADCSBB	%r9, %r8
	mov	%r8, (rp,n,8)
L(e01):
	jrcxz	L(end)
	lea	4(n), n
	jmp	L(top)
END (func)
