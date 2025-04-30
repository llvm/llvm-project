/* PowerPC64 __mpn_addmul_1 -- Multiply a limb vector with a limb and add
   the result to a second limb vector.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#ifdef USE_AS_SUBMUL
# define FUNC        __mpn_submul_1
# define ADDSUBC     subfe
# define ADDSUB      subfc
#else
# define FUNC        __mpn_addmul_1
# define ADDSUBC     adde
# define ADDSUB      addc
#endif

#define RP  r3
#define UP  r4
#define N   r5
#define VL  r6

#define R27SAVE  (-40)
#define R28SAVE  (-32)
#define R29SAVE  (-24)
#define R30SAVE  (-16)
#define R31SAVE  (-8)

ENTRY_TOCLESS (FUNC, 5)
	std	r31, R31SAVE(r1)
	rldicl.	r0, N, 0, 62
	std	r30, R30SAVE(r1)
	cmpdi	VL, r0, 2
	std	r29, R29SAVE(r1)
	addi	N, N, 3
	std	r28, R28SAVE(r1)
	srdi	N, N, 2
	std	r27, R27SAVE(r1)
	cfi_offset(r31, R31SAVE)
	cfi_offset(r30, R30SAVE)
	cfi_offset(r29, R29SAVE)
	cfi_offset(r28, R28SAVE)
	cfi_offset(r27, R27SAVE)
	mtctr	N
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	ld	r9, 0(UP)
	ld	r28, 0(RP)
	mulld	r0, r9, VL
	mulhdu	r12, r9, VL
	ADDSUB	r0, r0, r28
	std	r0, 0(RP)
	addi	RP, RP, 8
	ld	r9, 8(UP)
	ld	r27, 16(UP)
	addi	UP, UP, 24
#ifdef USE_AS_SUBMUL
	subfe	r11, r11, r11
#endif
	b	L(bot)

	.align	4
L(b00):	ld	r9, 0(UP)
	ld	r27, 8(UP)
	ld	r28, 0(RP)
	ld	r29, 8(RP)
	mulld	r0, r9, VL
	mulhdu	N, r9, VL
	mulld	r7, r27, VL
	mulhdu	r8, r27, VL
	addc	r7, r7, N
	addze	r12, r8
	ADDSUB	r0, r0, r28
	std	r0, 0(RP)
	ADDSUBC	r7, r7, r29
	std	r7, 8(RP)
	addi	RP, RP, 16
	ld	r9, 16(UP)
	ld	r27, 24(UP)
	addi	UP, UP, 32
#ifdef USE_AS_SUBMUL
	subfe	r11, r11, r11
#endif
	b	L(bot)

	.align	4
L(b01):	bdnz	L(gt1)
	ld	r9, 0(UP)
	ld	r11, 0(RP)
	mulld	r0, r9, VL
	mulhdu	r8, r9, VL
	ADDSUB	r0, r0, r11
	std	r0, 0(RP)
#ifdef USE_AS_SUBMUL
	subfe	r11, r11, r11
	addic	r11, r11, 1
#endif
	addze	RP, r8
	blr

L(gt1):	ld	r9, 0(UP)
	ld	r27, 8(UP)
	mulld	r0, r9, VL
	mulhdu	N, r9, VL
	mulld	r7, r27, VL
	mulhdu	r8, r27, VL
	ld	r9, 16(UP)
	ld	r28, 0(RP)
	ld	r29, 8(RP)
	ld	r30, 16(RP)
	mulld	r11, r9, VL
	mulhdu	r10, r9, VL
	addc	r7, r7, N
	adde	r11, r11, r8
	addze	r12, r10
	ADDSUB	r0, r0, r28
	std	r0, 0(RP)
	ADDSUBC	r7, r7, r29
	std	r7, 8(RP)
	ADDSUBC	r11, r11, r30
	std	r11, 16(RP)
	addi	RP, RP, 24
	ld	r9, 24(UP)
	ld	r27, 32(UP)
	addi	UP, UP, 40
#ifdef USE_AS_SUBMUL
	subfe	r11, r11, r11
#endif
	b	L(bot)

L(b10):	addic	r0, r0, r0
	li	r12, 0
	ld	r9, 0(UP)
	ld	r27, 8(UP)
	bdz	L(end)
	addi	UP, UP, 16

	.align	4
L(top):	mulld	r0, r9, VL
	mulhdu	N, r9, VL
	mulld	r7, r27, VL
	mulhdu	r8, r27, VL
	ld	r9, 0(UP)
	ld	r28, 0(RP)
	ld	r27, 8(UP)
	ld	r29, 8(RP)
	adde	r0, r0, r12
	adde	r7, r7, N
	mulld	N, r9, VL
	mulhdu	r10, r9, VL
	mulld	r11, r27, VL
	mulhdu	r12, r27, VL
	ld	r9, 16(UP)
	ld	r30, 16(RP)
	ld	r27, 24(UP)
	ld	r31, 24(RP)
	adde	N, N, r8
	adde	r11, r11, r10
	addze	r12, r12
	ADDSUB	r0, r0, r28
	std	r0, 0(RP)
	ADDSUBC	r7, r7, r29
	std	r7, 8(RP)
	ADDSUBC	N, N, r30
	std	N, 16(RP)
	ADDSUBC	r11, r11, r31
	std	r11, 24(RP)
	addi	UP, UP, 32
#ifdef USE_AS_SUBMUL
	subfe	r11, r11, r11
#endif
	addi	RP, RP, 32
L(bot):
#ifdef USE_AS_SUBMUL
	addic	r11, r11, 1
#endif
	bdnz	L(top)

L(end):	mulld	r0, r9, VL
	mulhdu	N, r9, VL
	mulld	r7, r27, VL
	mulhdu	r8, r27, VL
	ld	r28, 0(RP)
	ld	r29, 8(RP)
	adde	r0, r0, r12
	adde	r7, r7, N
	addze	r8, r8
	ADDSUB	r0, r0, r28
	std	r0, 0(RP)
	ADDSUBC	r7, r7, r29
	std	r7, 8(RP)
#ifdef USE_AS_SUBMUL
	subfe	r11, r11, r11
	addic	r11, r11, 1
#endif
	addze	RP, r8
	ld	r31, R31SAVE(r1)
	ld	r30, R30SAVE(r1)
	ld	r29, R29SAVE(r1)
	ld	r28, R28SAVE(r1)
	ld	r27, R27SAVE(r1)
	blr
END(FUNC)
