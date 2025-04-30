/* Optimized strcmp implementation for POWER7/PowerPC64.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#ifndef STRNCMP
# define STRNCMP strncmp
#endif

/* See strlen.s for comments on how the end-of-string testing works.  */

/* int [r3] strncmp (const char *s1 [r3],
		     const char *s2 [r4],
		     size_t size [r5])  */

	.machine power7
ENTRY_TOCLESS (STRNCMP, 5)
	CALL_MCOUNT 3

#define rTMP2	r0
#define rRTN	r3
#define rSTR1	r3	/* first string arg */
#define rSTR2	r4	/* second string arg */
#define rN	r5	/* max string length */
#define rWORD1	r6	/* current word in s1 */
#define rWORD2	r7	/* current word in s2 */
#define rWORD3  r10
#define rWORD4  r11
#define rFEFE	r8	/* constant 0xfefefefefefefeff (-0x0101010101010101) */
#define r7F7F	r9	/* constant 0x7f7f7f7f7f7f7f7f */
#define rNEG	r10	/* ~(word in s1 | 0x7f7f7f7f7f7f7f7f) */
#define rBITDIF	r11	/* bits that differ in s1 & s2 words */
#define rTMP	r12

	dcbt	0,rSTR1
	nop
	or	rTMP,rSTR2,rSTR1
	lis	r7F7F,0x7f7f
	dcbt	0,rSTR2
	nop
	clrldi.	rTMP,rTMP,61
	cmpldi	cr1,rN,0
	lis	rFEFE,-0x101
	bne	L(unaligned)
/* We are doubleword aligned so set up for two loops.  first a double word
   loop, then fall into the byte loop if any residual.  */
	srdi.	rTMP,rN,3
	clrldi	rN,rN,61
	addi	rFEFE,rFEFE,-0x101
	addi	r7F7F,r7F7F,0x7f7f
	cmpldi	cr1,rN,0
	beq	L(unaligned)

	mtctr	rTMP
	ld	rWORD1,0(rSTR1)
	ld	rWORD2,0(rSTR2)
	sldi	rTMP,rFEFE,32
	insrdi	r7F7F,r7F7F,32,0
	add	rFEFE,rFEFE,rTMP
	b	L(g1)

L(g0):
	ldu	rWORD1,8(rSTR1)
	bne	cr1,L(different)
	ldu	rWORD2,8(rSTR2)
L(g1):	add	rTMP,rFEFE,rWORD1
	nor	rNEG,r7F7F,rWORD1
	bdz	L(tail)
	and.	rTMP,rTMP,rNEG
	cmpd	cr1,rWORD1,rWORD2
	beq	L(g0)

/* OK. We've hit the end of the string. We need to be careful that
   we don't compare two strings as different because of gunk beyond
   the end of the strings...  */

#ifdef __LITTLE_ENDIAN__
L(endstring):
	addi    rTMP2, rTMP, -1
	beq	cr1, L(equal)
	andc    rTMP2, rTMP2, rTMP
	rldimi	rTMP2, rTMP2, 1, 0
	and	rWORD2, rWORD2, rTMP2	/* Mask off gunk.  */
	and	rWORD1, rWORD1, rTMP2
	cmpd	cr1, rWORD1, rWORD2
	beq	cr1, L(equal)
	cmpb	rBITDIF, rWORD1, rWORD2	/* 0xff on equal bytes.  */
	addi	rNEG, rBITDIF, 1
	orc	rNEG, rNEG, rBITDIF	/* 0's below LS differing byte.  */
	sldi	rNEG, rNEG, 8		/* 1's above LS differing byte.  */
	andc	rWORD1, rWORD1, rNEG	/* mask off MS bytes.  */
	andc	rWORD2, rWORD2, rNEG
	xor.	rBITDIF, rWORD1, rWORD2
	sub	rRTN, rWORD1, rWORD2
	blt	L(highbit)
	sradi	rRTN, rRTN, 63		/* must return an int.  */
	ori	rRTN, rRTN, 1
	blr
L(equal):
	li	rRTN, 0
	blr

L(different):
	ld	rWORD1, -8(rSTR1)
	cmpb	rBITDIF, rWORD1, rWORD2	/* 0xff on equal bytes.  */
	addi	rNEG, rBITDIF, 1
	orc	rNEG, rNEG, rBITDIF	/* 0's below LS differing byte.  */
	sldi	rNEG, rNEG, 8		/* 1's above LS differing byte.  */
	andc	rWORD1, rWORD1, rNEG	/* mask off MS bytes.  */
	andc	rWORD2, rWORD2, rNEG
	xor.	rBITDIF, rWORD1, rWORD2
	sub	rRTN, rWORD1, rWORD2
	blt	L(highbit)
	sradi	rRTN, rRTN, 63
	ori	rRTN, rRTN, 1
	blr
L(highbit):
	sradi	rRTN, rWORD2, 63
	ori	rRTN, rRTN, 1
	blr

#else
L(endstring):
	and	rTMP,r7F7F,rWORD1
	beq	cr1,L(equal)
	add	rTMP,rTMP,r7F7F
	xor.	rBITDIF,rWORD1,rWORD2
	andc	rNEG,rNEG,rTMP
	blt	L(highbit)
	cntlzd	rBITDIF,rBITDIF
	cntlzd	rNEG,rNEG
	addi	rNEG,rNEG,7
	cmpd	cr1,rNEG,rBITDIF
	sub	rRTN,rWORD1,rWORD2
	blt	cr1,L(equal)
	sradi	rRTN,rRTN,63		/* must return an int.  */
	ori	rRTN,rRTN,1
	blr
L(equal):
	li	rRTN,0
	blr

L(different):
	ld	rWORD1,-8(rSTR1)
	xor.	rBITDIF,rWORD1,rWORD2
	sub	rRTN,rWORD1,rWORD2
	blt	L(highbit)
	sradi	rRTN,rRTN,63
	ori	rRTN,rRTN,1
	blr
L(highbit):
	sradi	rRTN,rWORD2,63
	ori	rRTN,rRTN,1
	blr
#endif

/* Oh well.  In this case, we just do a byte-by-byte comparison.  */
	.align	4
L(tail):
	and.	rTMP,rTMP,rNEG
	cmpd	cr1,rWORD1,rWORD2
	bne	L(endstring)
	addi	rSTR1,rSTR1,8
	bne	cr1,L(different)
	addi	rSTR2,rSTR2,8
	cmpldi	cr1,rN,0
L(unaligned):
	mtctr	rN
	ble	cr1,L(ux)
L(uz):
	lbz	rWORD1,0(rSTR1)
	lbz	rWORD2,0(rSTR2)
	.align	4
L(u1):
	cmpdi	cr1,rWORD1,0
	bdz	L(u4)
	cmpd	rWORD1,rWORD2
	beq	cr1,L(u4)
	bne	L(u4)
	lbzu	rWORD3,1(rSTR1)
	lbzu	rWORD4,1(rSTR2)
	cmpdi	cr1,rWORD3,0
	bdz	L(u3)
	cmpd	rWORD3,rWORD4
	beq	cr1,L(u3)
	bne	L(u3)
	lbzu	rWORD1,1(rSTR1)
	lbzu	rWORD2,1(rSTR2)
	cmpdi	cr1,rWORD1,0
	bdz	L(u4)
	cmpd	rWORD1,rWORD2
	beq	cr1,L(u4)
	bne	L(u4)
	lbzu	rWORD3,1(rSTR1)
	lbzu	rWORD4,1(rSTR2)
	cmpdi	cr1,rWORD3,0
	bdz	L(u3)
	cmpd	rWORD3,rWORD4
	beq	cr1,L(u3)
	bne	L(u3)
	lbzu	rWORD1,1(rSTR1)
	lbzu	rWORD2,1(rSTR2)
	b	L(u1)

L(u3):  sub	rRTN,rWORD3,rWORD4
	blr
L(u4):	sub	rRTN,rWORD1,rWORD2
	blr
L(ux):
	li	rRTN,0
	blr
END (STRNCMP)
libc_hidden_builtin_def (strncmp)
