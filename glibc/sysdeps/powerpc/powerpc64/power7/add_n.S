/* PowerPC64 mpn_lshift --  mpn_add_n/mpn_sub_n -- mpn addition and
   subtraction.
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

/*             cycles/limb
 * POWER7             2.18
 */

#ifdef USE_AS_SUB
# define FUNC      __mpn_sub_n
# define ADDSUBC   subfe
#else
# define FUNC      __mpn_add_n
# define ADDSUBC   adde
#endif

#define RP  r3
#define UP  r4
#define VP  r5
#define N   r6

ENTRY_TOCLESS (FUNC, 5)
#ifdef USE_AS_SUB
	addic	r0, r1, -1
#else
	addic	r0, r0, 0
#endif
	andi.	r7, N, 1
	beq	L(bx0)

	ld	r7, 0(UP)
	ld	r9, 0(VP)
	ADDSUBC	r11, r9, r7
	std	r11, 0(RP)
	cmpldi	N, N, 1
	beq	N, L(end)
	addi	UP, UP, 8
	addi	VP, VP, 8
	addi	RP, RP, 8

L(bx0):	addi	r0, N, 2
	srdi	r0, r0, 2
	mtctr	r0

	andi.	r7, N, 2
	bne	L(mid)

	addi	UP, UP, 16
	addi	VP, VP, 16
	addi	RP, RP, 16

	.align	5
L(top):	ld	r6, -16(UP)
	ld	r7, -8(UP)
	ld	r8, -16(VP)
	ld	r9, -8(VP)
	ADDSUBC	r10, r8, N
	ADDSUBC	r11, r9, r7
	std	r10, -16(RP)
	std	r11, -8(RP)
L(mid):	ld	r6, 0(UP)
	ld	r7, 8(UP)
	ld	r8, 0(VP)
	ld	r9, 8(VP)
	ADDSUBC	r10, r8, N
	ADDSUBC	r11, r9, r7
	std	r10, 0(RP)
	std	r11, 8(RP)
	addi	UP, UP, 32
	addi	VP, VP, 32
	addi	RP, RP, 32
	bdnz	L(top)

L(end):	subfe	r3, r0, r0
#ifdef USE_AS_SUB
	neg	r3, r3
#else
	addi	r3, r3, 1
#endif
	blr
END(FUNC)
