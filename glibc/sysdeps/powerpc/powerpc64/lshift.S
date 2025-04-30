/* PowerPC64 mpn_lshift -- rp[] = up[] << cnt
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>

#define RP       r3
#define UP       r4
#define N        r5
#define CNT      r6

#define TNC      r0
#define U0      r30
#define U1      r31
#define U0SAVE  (-16)
#define U1SAVE  (-8)
#define RETVAL   r5

ENTRY_TOCLESS (__mpn_lshift, 5)
	std	U1, U1SAVE(r1)
	std	U0, U0SAVE(r1)
	cfi_offset(U1, U1SAVE)
	cfi_offset(U0, U0SAVE)
	subfic	TNC, CNT, 64
	sldi	r7, N, RP
	add	UP, UP, r7
	add	RP, RP, r7
	rldicl.	U0, N, 0, 62
	cmpdi	CNT, U0, 2
	addi	U1, N, RP
	ld	r10, -8(UP)
	srd	RETVAL, r10, TNC

	srdi	U1, U1, 2
	mtctr	U1
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	ld	r11, -16(UP)
	beq	cr6, L(b10)

	.align	4
L(b11):	sld	r8, r10, CNT
	srd	r9, r11, TNC
	ld	U1, -24(UP)
	addi	UP, UP, -24
	sld	r12, r11, CNT
	srd	r7, U1, TNC
	addi	RP, RP, 16
	bdnz	L(gt3)

	or	r11, r8, r9
	sld	r8, U1, CNT
	b	L(cj3)

	.align	4
L(gt3):	ld	U0, -8(UP)
	or	r11, r8, r9
	sld	r8, U1, CNT
	srd	r9, U0, TNC
	ld	U1, -16(UP)
	or	r10, r12, r7
	b	L(L11)

	.align	5
L(b10):	sld	r12, r10, CNT
	addi	RP, RP, 24
	srd	r7, r11, TNC
	bdnz	L(gt2)

	sld	r8, r11, CNT
	or	r10, r12, r7
	b	L(cj2)

L(gt2):	ld	U0, -24(UP)
	sld	r8, r11, CNT
	srd	r9, U0, TNC
	ld	U1, -32(UP)
	or	r10, r12, r7
	sld	r12, U0, CNT
	srd	r7, U1, 0
	ld	U0, -40(UP)
	or	r11, r8, r9
	addi	UP, UP, -16
	b	L(L10)

	.align	4
L(b00):	ld	U1, -16(UP)
	sld	r12, r10, CNT
	srd	r7, U1, TNC
	ld	U0, -24(UP)
	sld	r8, U1, CNT
	srd	r9, U0, TNC
	ld	U1, -32(UP)
	or	r10, r12, r7
	sld	r12, U0, CNT
	srd	r7, U1, TNC
	addi	RP, RP, r8
	bdz	L(cj4)

L(gt4):	addi	UP, UP, -32
	ld	U0, -8(UP)
	or	r11, r8, r9
	b	L(L00)

	.align	4
L(b01):	bdnz	L(gt1)
	sld	r8, r10, CNT
	std	r8, -8(RP)
	b	L(ret)

L(gt1):	ld	U0, -16(UP)
	sld	r8, r10, CNT
	srd	r9, U0, TNC
	ld	U1, -24(UP)
	sld	r12, U0, CNT
	srd	r7, U1, TNC
	ld	U0, -32(UP)
	or	r11, r8, r9
	sld	r8, U1, CNT
	srd	r9, U0, TNC
	ld	U1, -40(UP)
	addi	UP, UP, -40
	or	r10, r12, r7
	bdz	L(end)

	.align	5
L(top):	sld	r12, U0, CNT
	srd	r7, U1, TNC
	ld	U0, -8(UP)
	std	r11, -8(RP)
	or	r11, r8, r9
L(L00):	sld	r8, U1, CNT
	srd	r9, U0, TNC
	ld	U1, -16(UP)
	std	r10, -16(RP)
	or	r10, r12, r7
L(L11):	sld	r12, U0, CNT
	srd	r7, U1, TNC
	ld	U0, -24(UP)
	std	r11, -24(RP)
	or	r11, r8, r9
L(L10):	sld	r8, U1, CNT
	srd	r9, U0, TNC
	ld	U1, -32(UP)
	addi	UP, UP, -32
	std	r10, -32(RP)
	addi	RP, RP, -32
	or	r10, r12, r7
	bdnz	L(top)

	.align	5
L(end):	sld	r12, U0, CNT
	srd	r7, U1, TNC
	std	r11, -8(RP)
L(cj4):	or	r11, r8, r9
	sld	r8, U1, CNT
	std	r10, -16(RP)
L(cj3):	or	r10, r12, r7
	std	r11, -24(RP)
L(cj2):	std	r10, -32(RP)
	std	r8, -40(RP)

L(ret):	ld	U1, U1SAVE(r1)
	ld	U0, U0SAVE(r1)
	mr	RP, RETVAL
	blr
END(__mpn_lshift)
