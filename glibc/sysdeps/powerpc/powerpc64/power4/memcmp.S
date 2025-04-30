/* Optimized memcmp implementation for PowerPC64.
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

/* int [r3] memcmp (const char *s1 [r3],
		    const char *s2 [r4],
		    size_t size [r5])  */

#ifndef MEMCMP
# define MEMCMP memcmp
#endif

#ifndef __LITTLE_ENDIAN__
	.machine power4
#else
/* Little endian is only available since POWER8, so it's safe to
   specify .machine as power8 (or older), even though this is a POWER4
   file.  Since the little-endian code uses 'ldbrx', power7 is enough. */
	.machine power7
#endif
ENTRY_TOCLESS (MEMCMP, 4)
	CALL_MCOUNT 3

#define rRTN	r3
#define rSTR1	r3	/* first string arg */
#define rSTR2	r4	/* second string arg */
#define rN	r5	/* max string length */
#define rWORD1	r6	/* current word in s1 */
#define rWORD2	r7	/* current word in s2 */
#define rWORD3	r8	/* next word in s1 */
#define rWORD4	r9	/* next word in s2 */
#define rWORD5	r10	/* next word in s1 */
#define rWORD6	r11	/* next word in s2 */
#define rWORD7	r30	/* next word in s1 */
#define rWORD8	r31	/* next word in s2 */

	xor	r0, rSTR2, rSTR1
	cmpldi	cr6, rN, 0
	cmpldi	cr1, rN, 12
	clrldi.	r0, r0, 61
	clrldi	r12, rSTR1, 61
	cmpldi	cr5, r12, 0
	beq-	cr6, L(zeroLength)
	dcbt	0, rSTR1
	dcbt	0, rSTR2
/* If less than 8 bytes or not aligned, use the unaligned
   byte loop.  */
	blt	cr1, L(bytealigned)
	std	rWORD8, -8(r1)
	std	rWORD7, -16(r1)
	cfi_offset(rWORD8, -8)
	cfi_offset(rWORD7, -16)
	bne	L(unaligned)
/* At this point we know both strings have the same alignment and the
   compare length is at least 8 bytes.  r12 contains the low order
   3 bits of rSTR1 and cr5 contains the result of the logical compare
   of r12 to 0.  If r12 == 0 then we are already double word
   aligned and can perform the DW aligned loop.

   Otherwise we know the two strings have the same alignment (but not
   yet DW).  So we force the string addresses to the next lower DW
   boundary and special case this first DW using shift left to
   eliminate bits preceding the first byte.  Since we want to join the
   normal (DW aligned) compare loop, starting at the second double word,
   we need to adjust the length (rN) and special case the loop
   versioning for the first DW. This ensures that the loop count is
   correct and the first DW (shifted) is in the expected register pair.  */
	.align	4
L(samealignment):
	clrrdi	rSTR1, rSTR1, 3
	clrrdi	rSTR2, rSTR2, 3
	beq	cr5, L(DWaligned)
	add	rN, rN, r12
	sldi	rWORD6, r12, 3
	srdi	r0, rN, 5	/* Divide by 32 */
	andi.	r12, rN, 24	/* Get the DW remainder */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 0(rSTR1)
	ld	rWORD2, 0(rSTR2)
#endif
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	clrldi	rN, rN, 61
	beq	L(dPs4)
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
	bgt	cr1, L(dPs3)
	beq	cr1, L(dPs2)

/* Remainder is 8 */
	.align	3
L(dsP1):
	sld	rWORD5, rWORD1, rWORD6
	sld	rWORD6, rWORD2, rWORD6
	cmpld	cr5, rWORD5, rWORD6
	blt	cr7, L(dP1x)
/* Do something useful in this cycle since we have to branch anyway.  */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 8(rSTR1)
	ld	rWORD2, 8(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
	b	L(dP1e)
/* Remainder is 16 */
	.align	4
L(dPs2):
	sld	rWORD5, rWORD1, rWORD6
	sld	rWORD6, rWORD2, rWORD6
	cmpld	cr6, rWORD5, rWORD6
	blt	cr7, L(dP2x)
/* Do something useful in this cycle since we have to branch anyway.  */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD7, 8(rSTR1)
	ld	rWORD8, 8(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	b	L(dP2e)
/* Remainder is 24 */
	.align	4
L(dPs3):
	sld	rWORD3, rWORD1, rWORD6
	sld	rWORD4, rWORD2, rWORD6
	cmpld	cr1, rWORD3, rWORD4
	b	L(dP3e)
/* Count is a multiple of 32, remainder is 0 */
	.align	4
L(dPs4):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
	sld	rWORD1, rWORD1, rWORD6
	sld	rWORD2, rWORD2, rWORD6
	cmpld	cr7, rWORD1, rWORD2
	b	L(dP4e)

/* At this point we know both strings are double word aligned and the
   compare length is at least 8 bytes.  */
	.align	4
L(DWaligned):
	andi.	r12, rN, 24	/* Get the DW remainder */
	srdi	r0, rN, 5	/* Divide by 32 */
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	clrldi	rN, rN, 61
	beq	L(dP4)
	bgt	cr1, L(dP3)
	beq	cr1, L(dP2)

/* Remainder is 8 */
	.align	4
L(dP1):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
/* Normally we'd use rWORD7/rWORD8 here, but since we might exit early
   (8-15 byte compare), we want to use only volatile registers.  This
   means we can avoid restoring non-volatile registers since we did not
   change any on the early exit path.  The key here is the non-early
   exit path only cares about the condition code (cr5), not about which
   register pair was used.  */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 0(rSTR1)
	ld	rWORD6, 0(rSTR2)
#endif
	cmpld	cr5, rWORD5, rWORD6
	blt	cr7, L(dP1x)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 8(rSTR1)
	ld	rWORD2, 8(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
L(dP1e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 16(rSTR1)
	ld	rWORD4, 16(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 24(rSTR1)
	ld	rWORD6, 24(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(dLcr5x)
	bne	cr7, L(dLcr7x)

#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ldu	rWORD7, 32(rSTR1)
	ldu	rWORD8, 32(rSTR2)
#endif
	bne	cr1, L(dLcr1)
	cmpld	cr5, rWORD7, rWORD8
	bdnz	L(dLoop)
	bne	cr6, L(dLcr6)
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
	.align	3
L(dP1x):
	sldi.	r12, rN, 3
	bne	cr5, L(dLcr5x)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	bne	L(d00)
	li	rRTN, 0
	blr

/* Remainder is 16 */
	.align	4
L(dP2):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 0(rSTR1)
	ld	rWORD6, 0(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	blt	cr7, L(dP2x)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD7, 8(rSTR1)
	ld	rWORD8, 8(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
L(dP2e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 16(rSTR1)
	ld	rWORD2, 16(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 24(rSTR1)
	ld	rWORD4, 24(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#endif
	bne	cr6, L(dLcr6)
	bne	cr5, L(dLcr5)
	b	L(dLoop2)
/* Again we are on a early exit path (16-23 byte compare), we want to
   only use volatile registers and avoid restoring non-volatile
   registers.  */
	.align	4
L(dP2x):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 8(rSTR1)
	ld	rWORD4, 8(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
	sldi.	r12, rN, 3
	bne	cr6, L(dLcr6x)
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#endif
	bne	cr1, L(dLcr1x)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	bne	L(d00)
	li	rRTN, 0
	blr

/* Remainder is 24 */
	.align	4
L(dP3):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 0(rSTR1)
	ld	rWORD4, 0(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
L(dP3e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 8(rSTR1)
	ld	rWORD6, 8(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	blt	cr7, L(dP3x)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD7, 16(rSTR1)
	ld	rWORD8, 16(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 24(rSTR1)
	ld	rWORD2, 24(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
#endif
	bne	cr1, L(dLcr1)
	bne	cr6, L(dLcr6)
	b	L(dLoop1)
/* Again we are on a early exit path (24-31 byte compare), we want to
   only use volatile registers and avoid restoring non-volatile
   registers.  */
	.align	4
L(dP3x):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 16(rSTR1)
	ld	rWORD2, 16(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
	sldi.	r12, rN, 3
	bne	cr1, L(dLcr1x)
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
#endif
	bne	cr6, L(dLcr6x)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	bne	cr7, L(dLcr7x)
	bne	L(d00)
	li	rRTN, 0
	blr

/* Count is a multiple of 32, remainder is 0 */
	.align	4
L(dP4):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 0(rSTR1)
	ld	rWORD2, 0(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
L(dP4e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 8(rSTR1)
	ld	rWORD4, 8(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 16(rSTR1)
	ld	rWORD6, 16(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ldu	rWORD7, 24(rSTR1)
	ldu	rWORD8, 24(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	bne	cr7, L(dLcr7)
	bne	cr1, L(dLcr1)
	bdz-	L(d24)		/* Adjust CTR as we start with +4 */
/* This is the primary loop */
	.align	4
L(dLoop):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 8(rSTR1)
	ld	rWORD2, 8(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
	bne	cr6, L(dLcr6)
L(dLoop1):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 16(rSTR1)
	ld	rWORD4, 16(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(dLcr5)
L(dLoop2):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 24(rSTR1)
	ld	rWORD6, 24(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	bne	cr7, L(dLcr7)
L(dLoop3):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ldu	rWORD7, 32(rSTR1)
	ldu	rWORD8, 32(rSTR2)
#endif
	bne-	cr1, L(dLcr1)
	cmpld	cr7, rWORD1, rWORD2
	bdnz+	L(dLoop)

L(dL4):
	cmpld	cr1, rWORD3, rWORD4
	bne	cr6, L(dLcr6)
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(dLcr5)
	cmpld	cr5, rWORD7, rWORD8
L(d44):
	bne	cr7, L(dLcr7)
L(d34):
	bne	cr1, L(dLcr1)
L(d24):
	bne	cr6, L(dLcr6)
L(d14):
	sldi.	r12, rN, 3
	bne	cr5, L(dLcr5)
L(d04):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	beq	L(zeroLength)
/* At this point we have a remainder of 1 to 7 bytes to compare.  Since
   we are aligned it is safe to load the whole double word, and use
   shift right double to eliminate bits beyond the compare length.  */
L(d00):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 8(rSTR1)
	ld	rWORD2, 8(rSTR2)
#endif
	srd	rWORD1, rWORD1, rN
	srd	rWORD2, rWORD2, rN
	cmpld	cr7, rWORD1, rWORD2
	bne	cr7, L(dLcr7x)
	li	rRTN, 0
	blr

	.align	4
L(dLcr7):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
L(dLcr7x):
	li	rRTN, 1
	bgtlr	cr7
	li	rRTN, -1
	blr
	.align	4
L(dLcr1):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
L(dLcr1x):
	li	rRTN, 1
	bgtlr	cr1
	li	rRTN, -1
	blr
	.align	4
L(dLcr6):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
L(dLcr6x):
	li	rRTN, 1
	bgtlr	cr6
	li	rRTN, -1
	blr
	.align	4
L(dLcr5):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
L(dLcr5x):
	li	rRTN, 1
	bgtlr	cr5
	li	rRTN, -1
	blr

	.align	4
L(bytealigned):
	mtctr	rN	/* Power4 wants mtctr 1st in dispatch group */
#if 0
/* Huh?  We've already branched on cr6!  */
	beq-	cr6, L(zeroLength)
#endif

/* We need to prime this loop.  This loop is swing modulo scheduled
   to avoid pipe delays.  The dependent instruction latencies (load to
   compare to conditional branch) is 2 to 3 cycles.  In this loop each
   dispatch group ends in a branch and takes 1 cycle.  Effectively
   the first iteration of the loop only serves to load operands and
   branches based on compares are delayed until the next loop.

   So we must precondition some registers and condition codes so that
   we don't exit the loop early on the first iteration.  */

	lbz	rWORD1, 0(rSTR1)
	lbz	rWORD2, 0(rSTR2)
	bdz-	L(b11)
	cmpld	cr7, rWORD1, rWORD2
	lbz	rWORD3, 1(rSTR1)
	lbz	rWORD4, 1(rSTR2)
	bdz-	L(b12)
	cmpld	cr1, rWORD3, rWORD4
	lbzu	rWORD5, 2(rSTR1)
	lbzu	rWORD6, 2(rSTR2)
	bdz-	L(b13)
	.align	4
L(bLoop):
	lbzu	rWORD1, 1(rSTR1)
	lbzu	rWORD2, 1(rSTR2)
	bne-	cr7, L(bLcr7)

	cmpld	cr6, rWORD5, rWORD6
	bdz-	L(b3i)

	lbzu	rWORD3, 1(rSTR1)
	lbzu	rWORD4, 1(rSTR2)
	bne-	cr1, L(bLcr1)

	cmpld	cr7, rWORD1, rWORD2
	bdz-	L(b2i)

	lbzu	rWORD5, 1(rSTR1)
	lbzu	rWORD6, 1(rSTR2)
	bne-	cr6, L(bLcr6)

	cmpld	cr1, rWORD3, rWORD4
	bdnz+	L(bLoop)

/* We speculatively loading bytes before we have tested the previous
   bytes.  But we must avoid overrunning the length (in the ctr) to
   prevent these speculative loads from causing a segfault.  In this
   case the loop will exit early (before the all pending bytes are
   tested.  In this case we must complete the pending operations
   before returning.  */
L(b1i):
	bne-	cr7, L(bLcr7)
	bne-	cr1, L(bLcr1)
	b	L(bx56)
	.align	4
L(b2i):
	bne-	cr6, L(bLcr6)
	bne-	cr7, L(bLcr7)
	b	L(bx34)
	.align	4
L(b3i):
	bne-	cr1, L(bLcr1)
	bne-	cr6, L(bLcr6)
	b	L(bx12)
	.align	4
L(bLcr7):
	li	rRTN, 1
	bgtlr	cr7
	li	rRTN, -1
	blr
L(bLcr1):
	li	rRTN, 1
	bgtlr	cr1
	li	rRTN, -1
	blr
L(bLcr6):
	li	rRTN, 1
	bgtlr	cr6
	li	rRTN, -1
	blr

L(b13):
	bne-	cr7, L(bx12)
	bne-	cr1, L(bx34)
L(bx56):
	sub	rRTN, rWORD5, rWORD6
	blr
	nop
L(b12):
	bne-	cr7, L(bx12)
L(bx34):
	sub	rRTN, rWORD3, rWORD4
	blr
L(b11):
L(bx12):
	sub	rRTN, rWORD1, rWORD2
	blr
	.align	4
L(zeroLength):
	li	rRTN, 0
	blr

	.align	4
/* At this point we know the strings have different alignment and the
   compare length is at least 8 bytes.  r12 contains the low order
   3 bits of rSTR1 and cr5 contains the result of the logical compare
   of r12 to 0.  If r12 == 0 then rStr1 is double word
   aligned and can perform the DWunaligned loop.

   Otherwise we know that rSTR1 is not already DW aligned yet.
   So we can force the string addresses to the next lower DW
   boundary and special case this first DW using shift left to
   eliminate bits preceding the first byte.  Since we want to join the
   normal (DWaligned) compare loop, starting at the second double word,
   we need to adjust the length (rN) and special case the loop
   versioning for the first DW. This ensures that the loop count is
   correct and the first DW (shifted) is in the expected resister pair.  */
#define rSHL		r29	/* Unaligned shift left count.  */
#define rSHR		r28	/* Unaligned shift right count.  */
#define rWORD8_SHIFT	r27	/* Left rotation temp for rWORD2.  */
#define rWORD2_SHIFT	r26	/* Left rotation temp for rWORD4.  */
#define rWORD4_SHIFT	r25	/* Left rotation temp for rWORD6.  */
#define rWORD6_SHIFT	r24	/* Left rotation temp for rWORD8.  */
L(unaligned):
	std	rSHL, -24(r1)
	cfi_offset(rSHL, -24)
	clrldi	rSHL, rSTR2, 61
	beq-	cr6, L(duzeroLength)
	std	rSHR, -32(r1)
	cfi_offset(rSHR, -32)
	beq	cr5, L(DWunaligned)
	std	rWORD8_SHIFT, -40(r1)
	cfi_offset(rWORD8_SHIFT, -40)
/* Adjust the logical start of rSTR2 to compensate for the extra bits
   in the 1st rSTR1 DW.  */
	sub	rWORD8_SHIFT, rSTR2, r12
/* But do not attempt to address the DW before that DW that contains
   the actual start of rSTR2.  */
	clrrdi	rSTR2, rSTR2, 3
	std	rWORD2_SHIFT, -48(r1)
/* Compute the left/right shift counts for the unaligned rSTR2,
   compensating for the logical (DW aligned) start of rSTR1.  */
	clrldi	rSHL, rWORD8_SHIFT, 61
	clrrdi	rSTR1, rSTR1, 3
	std	rWORD4_SHIFT, -56(r1)
	sldi	rSHL, rSHL, 3
	cmpld	cr5, rWORD8_SHIFT, rSTR2
	add	rN, rN, r12
	sldi	rWORD6, r12, 3
	std	rWORD6_SHIFT, -64(r1)
	cfi_offset(rWORD2_SHIFT, -48)
	cfi_offset(rWORD4_SHIFT, -56)
	cfi_offset(rWORD6_SHIFT, -64)
	subfic	rSHR, rSHL, 64
	srdi	r0, rN, 5	/* Divide by 32 */
	andi.	r12, rN, 24	/* Get the DW remainder */
/* We normally need to load 2 DWs to start the unaligned rSTR2, but in
   this special case those bits may be discarded anyway.  Also we
   must avoid loading a DW where none of the bits are part of rSTR2 as
   this may cross a page boundary and cause a page fault.  */
	li	rWORD8, 0
	blt	cr5, L(dus0)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD8, 0(rSTR2)
	addi	rSTR2, rSTR2, 8
#endif
	sld	rWORD8, rWORD8, rSHL

L(dus0):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 0(rSTR1)
	ld	rWORD2, 0(rSTR2)
#endif
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	srd	r12, rWORD2, rSHR
	clrldi	rN, rN, 61
	beq	L(duPs4)
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
	or	rWORD8, r12, rWORD8
	bgt	cr1, L(duPs3)
	beq	cr1, L(duPs2)

/* Remainder is 8 */
	.align	4
L(dusP1):
	sld	rWORD8_SHIFT, rWORD2, rSHL
	sld	rWORD7, rWORD1, rWORD6
	sld	rWORD8, rWORD8, rWORD6
	bge	cr7, L(duP1e)
/* At this point we exit early with the first double word compare
   complete and remainder of 0 to 7 bytes.  See L(du14) for details on
   how we handle the remaining bytes.  */
	cmpld	cr5, rWORD7, rWORD8
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD2, 8(rSTR2)
#endif
	srd	r0, rWORD2, rSHR
	b	L(dutrim)
/* Remainder is 16 */
	.align	4
L(duPs2):
	sld	rWORD6_SHIFT, rWORD2, rSHL
	sld	rWORD5, rWORD1, rWORD6
	sld	rWORD6, rWORD8, rWORD6
	b	L(duP2e)
/* Remainder is 24 */
	.align	4
L(duPs3):
	sld	rWORD4_SHIFT, rWORD2, rSHL
	sld	rWORD3, rWORD1, rWORD6
	sld	rWORD4, rWORD8, rWORD6
	b	L(duP3e)
/* Count is a multiple of 32, remainder is 0 */
	.align	4
L(duPs4):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
	or	rWORD8, r12, rWORD8
	sld	rWORD2_SHIFT, rWORD2, rSHL
	sld	rWORD1, rWORD1, rWORD6
	sld	rWORD2, rWORD8, rWORD6
	b	L(duP4e)

/* At this point we know rSTR1 is double word aligned and the
   compare length is at least 8 bytes.  */
	.align	4
L(DWunaligned):
	std	rWORD8_SHIFT, -40(r1)
	clrrdi	rSTR2, rSTR2, 3
	std	rWORD2_SHIFT, -48(r1)
	srdi	r0, rN, 5	/* Divide by 32 */
	std	rWORD4_SHIFT, -56(r1)
	andi.	r12, rN, 24	/* Get the DW remainder */
	std	rWORD6_SHIFT, -64(r1)
	cfi_offset(rWORD8_SHIFT, -40)
	cfi_offset(rWORD2_SHIFT, -48)
	cfi_offset(rWORD4_SHIFT, -56)
	cfi_offset(rWORD6_SHIFT, -64)
	sldi	rSHL, rSHL, 3
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR2, rSTR2, 8
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD6, 0(rSTR2)
	ldu	rWORD8, 8(rSTR2)
#endif
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	clrldi	rN, rN, 61
	subfic	rSHR, rSHL, 64
	sld	rWORD6_SHIFT, rWORD6, rSHL
	beq	L(duP4)
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
	bgt	cr1, L(duP3)
	beq	cr1, L(duP2)

/* Remainder is 8 */
	.align	4
L(duP1):
	srd	r12, rWORD8, rSHR
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	addi	rSTR1, rSTR1, 8
#else
	ld	rWORD7, 0(rSTR1)
#endif
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	blt	cr7, L(duP1x)
L(duP1e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 8(rSTR1)
	ld	rWORD2, 8(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 16(rSTR1)
	ld	rWORD4, 16(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	bne	cr5, L(duLcr5)
	or	rWORD4, r12, rWORD2_SHIFT
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 24(rSTR1)
	ld	rWORD6, 24(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	bne	cr7, L(duLcr7)
	or	rWORD6, r0, rWORD4_SHIFT
	cmpld	cr6, rWORD5, rWORD6
	b	L(duLoop3)
	.align	4
/* At this point we exit early with the first double word compare
   complete and remainder of 0 to 7 bytes.  See L(du14) for details on
   how we handle the remaining bytes.  */
L(duP1x):
	cmpld	cr5, rWORD7, rWORD8
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD2, 8(rSTR2)
#endif
	srd	r0, rWORD2, rSHR
	b	L(dutrim)
/* Remainder is 16 */
	.align	4
L(duP2):
	srd	r0, rWORD8, rSHR
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	addi	rSTR1, rSTR1, 8
#else
	ld	rWORD5, 0(rSTR1)
#endif
	or	rWORD6, r0, rWORD6_SHIFT
	sld	rWORD6_SHIFT, rWORD8, rSHL
L(duP2e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD7, 8(rSTR1)
	ld	rWORD8, 8(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	blt	cr7, L(duP2x)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 16(rSTR1)
	ld	rWORD2, 16(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	bne	cr6, L(duLcr6)
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 24(rSTR1)
	ld	rWORD4, 24(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
	bne	cr5, L(duLcr5)
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	or	rWORD4, r12, rWORD2_SHIFT
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#endif
	cmpld	cr1, rWORD3, rWORD4
	b	L(duLoop2)
	.align	4
L(duP2x):
	cmpld	cr5, rWORD7, rWORD8
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#endif
	bne	cr6, L(duLcr6)
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD2, 8(rSTR2)
#endif
	srd	r0, rWORD2, rSHR
	b	L(dutrim)

/* Remainder is 24 */
	.align	4
L(duP3):
	srd	r12, rWORD8, rSHR
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	addi	rSTR1, rSTR1, 8
#else
	ld	rWORD3, 0(rSTR1)
#endif
	sld	rWORD4_SHIFT, rWORD8, rSHL
	or	rWORD4, r12, rWORD6_SHIFT
L(duP3e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 8(rSTR1)
	ld	rWORD6, 8(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	or	rWORD6, r0, rWORD4_SHIFT
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD7, 16(rSTR1)
	ld	rWORD8, 16(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	bne	cr1, L(duLcr1)
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	blt	cr7, L(duP3x)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 24(rSTR1)
	ld	rWORD2, 24(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	bne	cr6, L(duLcr6)
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
#endif
	cmpld	cr7, rWORD1, rWORD2
	b	L(duLoop1)
	.align	4
L(duP3x):
#ifndef __LITTLE_ENDIAN__
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
#endif
#if 0
/* Huh?  We've already branched on cr1!  */
	bne	cr1, L(duLcr1)
#endif
	cmpld	cr5, rWORD7, rWORD8
	bne	cr6, L(duLcr6)
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD2, 8(rSTR2)
#endif
	srd	r0, rWORD2, rSHR
	b	L(dutrim)

/* Count is a multiple of 32, remainder is 0 */
	.align	4
L(duP4):
	mtctr	r0	/* Power4 wants mtctr 1st in dispatch group */
	srd	r0, rWORD8, rSHR
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	addi	rSTR1, rSTR1, 8
#else
	ld	rWORD1, 0(rSTR1)
#endif
	sld	rWORD2_SHIFT, rWORD8, rSHL
	or	rWORD2, r0, rWORD6_SHIFT
L(duP4e):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 8(rSTR1)
	ld	rWORD4, 8(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	or	rWORD4, r12, rWORD2_SHIFT
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 16(rSTR1)
	ld	rWORD6, 16(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
	bne	cr7, L(duLcr7)
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	or	rWORD6, r0, rWORD4_SHIFT
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ldu	rWORD7, 24(rSTR1)
	ldu	rWORD8, 24(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	bne	cr1, L(duLcr1)
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	cmpld	cr5, rWORD7, rWORD8
	bdz-	L(du24)		/* Adjust CTR as we start with +4 */
/* This is the primary loop */
	.align	4
L(duLoop):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD1, 8(rSTR1)
	ld	rWORD2, 8(rSTR2)
#endif
	cmpld	cr1, rWORD3, rWORD4
	bne	cr6, L(duLcr6)
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
L(duLoop1):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD3, 0, rSTR1
	ldbrx	rWORD4, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD3, 16(rSTR1)
	ld	rWORD4, 16(rSTR2)
#endif
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(duLcr5)
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	or	rWORD4, r12, rWORD2_SHIFT
L(duLoop2):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD5, 0, rSTR1
	ldbrx	rWORD6, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD5, 24(rSTR1)
	ld	rWORD6, 24(rSTR2)
#endif
	cmpld	cr5, rWORD7, rWORD8
	bne	cr7, L(duLcr7)
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	or	rWORD6, r0, rWORD4_SHIFT
L(duLoop3):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD7, 0, rSTR1
	ldbrx	rWORD8, 0, rSTR2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
#else
	ldu	rWORD7, 32(rSTR1)
	ldu	rWORD8, 32(rSTR2)
#endif
	cmpld	cr7, rWORD1, rWORD2
	bne-	cr1, L(duLcr1)
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	bdnz+	L(duLoop)

L(duL4):
#if 0
/* Huh?  We've already branched on cr1!  */
	bne	cr1, L(duLcr1)
#endif
	cmpld	cr1, rWORD3, rWORD4
	bne	cr6, L(duLcr6)
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(duLcr5)
	cmpld	cr5, rWORD7, rWORD8
L(du44):
	bne	cr7, L(duLcr7)
L(du34):
	bne	cr1, L(duLcr1)
L(du24):
	bne	cr6, L(duLcr6)
L(du14):
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
/* At this point we have a remainder of 1 to 7 bytes to compare.  We use
   shift right double to eliminate bits beyond the compare length.

   However it may not be safe to load rWORD2 which may be beyond the
   string length. So we compare the bit length of the remainder to
   the right shift count (rSHR). If the bit count is less than or equal
   we do not need to load rWORD2 (all significant bits are already in
   rWORD8_SHIFT).  */
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD2, 0, rSTR2
	addi	rSTR2, rSTR2, 8
#else
	ld	rWORD2, 8(rSTR2)
#endif
	srd	r0, rWORD2, rSHR
	.align	4
L(dutrim):
#ifdef __LITTLE_ENDIAN__
	ldbrx	rWORD1, 0, rSTR1
#else
	ld	rWORD1, 8(rSTR1)
#endif
	ld	rWORD8, -8(r1)
	subfic	rN, rN, 64	/* Shift count is 64 - (rN * 8).  */
	or	rWORD2, r0, rWORD8_SHIFT
	ld	rWORD7, -16(r1)
	ld	rSHL, -24(r1)
	srd	rWORD1, rWORD1, rN
	srd	rWORD2, rWORD2, rN
	ld	rSHR, -32(r1)
	ld	rWORD8_SHIFT, -40(r1)
	li	rRTN, 0
	cmpld	cr7, rWORD1, rWORD2
	ld	rWORD2_SHIFT, -48(r1)
	ld	rWORD4_SHIFT, -56(r1)
	beq	cr7, L(dureturn24)
	li	rRTN, 1
	ld	rWORD6_SHIFT, -64(r1)
	bgtlr	cr7
	li	rRTN, -1
	blr
	.align	4
L(duLcr7):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
	li	rRTN, 1
	bgt	cr7, L(dureturn29)
	ld	rSHL, -24(r1)
	ld	rSHR, -32(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	4
L(duLcr1):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
	li	rRTN, 1
	bgt	cr1, L(dureturn29)
	ld	rSHL, -24(r1)
	ld	rSHR, -32(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	4
L(duLcr6):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
	li	rRTN, 1
	bgt	cr6, L(dureturn29)
	ld	rSHL, -24(r1)
	ld	rSHR, -32(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	4
L(duLcr5):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
	li	rRTN, 1
	bgt	cr5, L(dureturn29)
	ld	rSHL, -24(r1)
	ld	rSHR, -32(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	3
L(duZeroReturn):
	li	rRTN, 0
	.align	4
L(dureturn):
	ld	rWORD8, -8(r1)
	ld	rWORD7, -16(r1)
L(dureturn29):
	ld	rSHL, -24(r1)
	ld	rSHR, -32(r1)
L(dureturn27):
	ld	rWORD8_SHIFT, -40(r1)
L(dureturn26):
	ld	rWORD2_SHIFT, -48(r1)
L(dureturn25):
	ld	rWORD4_SHIFT, -56(r1)
L(dureturn24):
	ld	rWORD6_SHIFT, -64(r1)
	blr
L(duzeroLength):
	li	rRTN, 0
	blr

END (MEMCMP)
libc_hidden_builtin_def (memcmp)
weak_alias (memcmp, bcmp)
