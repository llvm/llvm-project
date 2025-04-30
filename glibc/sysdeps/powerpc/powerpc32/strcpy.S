/* Optimized strcpy implementation for PowerPC.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* See strlen.s for comments on how the end-of-string testing works.  */

/* char * [r3] strcpy (char *dest [r3], const char *src [r4])  */

EALIGN (strcpy, 4, 0)

#define rTMP	r0
#define rRTN	r3	/* incoming DEST arg preserved as result */
#define rSRC	r4	/* pointer to previous word in src */
#define rDEST	r5	/* pointer to previous word in dest */
#define rWORD	r6	/* current word from src */
#define rFEFE	r7	/* constant 0xfefefeff (-0x01010101) */
#define r7F7F	r8	/* constant 0x7f7f7f7f */
#define rNEG	r9	/* ~(word in s1 | 0x7f7f7f7f) */
#define rALT	r10	/* alternate word from src */


	or	rTMP, rSRC, rRTN
	clrlwi.	rTMP, rTMP, 30
	addi	rDEST, rRTN, -4
	bne	L(unaligned)

	lis	rFEFE, -0x101
	lis	r7F7F, 0x7f7f
	lwz	rWORD, 0(rSRC)
	addi	rFEFE, rFEFE, -0x101
	addi	r7F7F, r7F7F, 0x7f7f
	b	L(g2)

L(g0):	lwzu	rALT, 4(rSRC)
	stwu	rWORD, 4(rDEST)
	add	rTMP, rFEFE, rALT
	nor	rNEG, r7F7F, rALT
	and.	rTMP, rTMP, rNEG
	bne-	L(g1)
	lwzu	rWORD, 4(rSRC)
	stwu	rALT, 4(rDEST)
L(g2):	add	rTMP, rFEFE, rWORD
	nor	rNEG, r7F7F, rWORD
	and.	rTMP, rTMP, rNEG
	beq+	L(g0)

	mr	rALT, rWORD
/* We've hit the end of the string.  Do the rest byte-by-byte.  */
L(g1):
#ifdef __LITTLE_ENDIAN__
	rlwinm.	rTMP, rALT, 0, 24, 31
	stb	rALT, 4(rDEST)
	beqlr-
	rlwinm.	rTMP, rALT, 24, 24, 31
	stb	rTMP, 5(rDEST)
	beqlr-
	rlwinm.	rTMP, rALT, 16, 24, 31
	stb	rTMP, 6(rDEST)
	beqlr-
	rlwinm	rTMP, rALT, 8, 24, 31
	stb	rTMP, 7(rDEST)
	blr
#else
	rlwinm.	rTMP, rALT, 8, 24, 31
	stb	rTMP, 4(rDEST)
	beqlr-
	rlwinm.	rTMP, rALT, 16, 24, 31
	stb	rTMP, 5(rDEST)
	beqlr-
	rlwinm.	rTMP, rALT, 24, 24, 31
	stb	rTMP, 6(rDEST)
	beqlr-
	stb	rALT, 7(rDEST)
	blr
#endif

/* Oh well.  In this case, we just do a byte-by-byte copy.  */
	.align 4
	nop
L(unaligned):
	lbz	rWORD, 0(rSRC)
	addi	rDEST, rRTN, -1
	cmpwi	rWORD, 0
	beq-	L(u2)

L(u0):	lbzu	rALT, 1(rSRC)
	stbu	rWORD, 1(rDEST)
	cmpwi	rALT, 0
	beq-	L(u1)
	nop		/* Let 601 load start of loop.  */
	lbzu	rWORD, 1(rSRC)
	stbu	rALT, 1(rDEST)
	cmpwi	rWORD, 0
	bne+	L(u0)
L(u2):	stb	rWORD, 1(rDEST)
	blr
L(u1):	stb	rALT, 1(rDEST)
	blr

END (strcpy)
libc_hidden_builtin_def (strcpy)
