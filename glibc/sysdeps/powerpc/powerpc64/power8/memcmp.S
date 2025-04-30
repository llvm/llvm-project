/* Optimized memcmp implementation for POWER7/PowerPC64.
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

/* int [r3] memcmp (const char *s1 [r3],
		    const char *s2 [r4],
		    size_t size [r5])  */

#ifndef MEMCMP
# define MEMCMP memcmp
#endif
	.machine  power8
ENTRY_TOCLESS (MEMCMP, 4)
	CALL_MCOUNT 3

#define rRTN		r3
#define rSTR1		r3	/* First string arg.  */
#define rSTR2		r4	/* Second string arg.  */
#define rN		r5	/* Max string length.  */
#define rWORD1		r6	/* Current word in s1.  */
#define rWORD2		r7	/* Current word in s2.  */
#define rWORD3		r8	/* Next word in s1.  */
#define rWORD4		r9	/* Next word in s2.  */
#define rWORD5		r10	/* Next word in s1.  */
#define rWORD6		r11	/* Next word in s2.  */

#define rOFF8		r20	/* 8 bytes offset.  */
#define rOFF16  	r21	/* 16 bytes offset.  */
#define rOFF24		r22	/* 24 bytes offset.  */
#define rOFF32		r23	/* 24 bytes offset.  */
#define rWORD6_SHIFT	r24	/* Left rotation temp for rWORD8.  */
#define rWORD4_SHIFT	r25	/* Left rotation temp for rWORD6.  */
#define rWORD2_SHIFT	r26	/* Left rotation temp for rWORD4.  */
#define rWORD8_SHIFT	r27	/* Left rotation temp for rWORD2.  */
#define rSHR		r28	/* Unaligned shift right count.  */
#define rSHL		r29	/* Unaligned shift left count.  */
#define rWORD7		r30	/* Next word in s1.  */
#define rWORD8		r31	/* Next word in s2.  */

#define rWORD8SAVE	(-8)
#define rWORD7SAVE	(-16)
#define rOFF8SAVE	(-24)
#define rOFF16SAVE	(-32)
#define rOFF24SAVE	(-40)
#define rOFF32SAVE	(-48)
#define rSHRSAVE	(-56)
#define rSHLSAVE	(-64)
#define rWORD8SHIFTSAVE	(-72)
#define rWORD2SHIFTSAVE	(-80)
#define rWORD4SHIFTSAVE	(-88)
#define rWORD6SHIFTSAVE	(-96)

#ifdef __LITTLE_ENDIAN__
# define LD	ldbrx
#else
# define LD	ldx
#endif

	xor	r10, rSTR2, rSTR1
	cmpldi	cr6, rN, 0
	cmpldi	cr1, rN, 8
	clrldi.	r0, r10, 61
	clrldi	r12, rSTR1, 61
	cmpldi	cr5, r12, 0
	beq-	cr6, L(zeroLength)
	dcbt	0, rSTR1
	dcbt	0, rSTR2
	/* If less than 8 bytes or not aligned, use the unaligned
	   byte loop.  */
	blt	cr1, L(bytealigned)
	bne	L(unalignedqw)
/* At this point we know both strings have the same alignment and the
   compare length is at least 8 bytes.  r12 contains the low order
   3 bits of rSTR1 and cr5 contains the result of the logical compare
   of r12 to 0.  If r12 == 0 then we are already double word
   aligned and can perform the DW aligned loop.  */

	.align	4
L(samealignment):
	or	r11, rSTR2, rSTR1
	clrldi.	r11, r11, 60
	beq	L(qw_align)
	/* Try to align to QW else proceed to DW loop.  */
	clrldi.	r10, r10, 60
	bne	L(DW)
	/* For the difference to reach QW alignment, load as DW.  */
	clrrdi	rSTR1, rSTR1, 3
	clrrdi	rSTR2, rSTR2, 3
	subfic	r10, r12, 8
	LD	rWORD1, 0, rSTR1
	LD	rWORD2, 0, rSTR2
	sldi	r9, r10, 3
	subfic	r9, r9, 64
	sld	rWORD1, rWORD1, r9
	sld	rWORD2, rWORD2, r9
	cmpld	cr6, rWORD1, rWORD2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
	bne	cr6, L(ret_diff)
	subf	rN, r10, rN

	cmpld	cr6, r11, r12
	bgt	cr6, L(qw_align)
	LD	rWORD1, 0, rSTR1
	LD	rWORD2, 0, rSTR2
	cmpld	cr6, rWORD1, rWORD2
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
	bne	cr6, L(different)
	cmpldi	cr6, rN, 8
	ble	cr6, L(zeroLength)
	addi	rN, rN, -8
	/* Now both rSTR1 and rSTR2 are aligned to QW.  */
	.align	4
L(qw_align):
	vspltisb	v0, 0
	srdi.	r6, rN, 6
	li	r8, 16
	li	r10, 32
	li	r11, 48
	ble	cr0, L(lessthan64)
	mtctr	r6
	vspltisb	v8, 0
	vspltisb	v6, 0
	/* Aligned vector loop.  */
	.align	4
L(aligned_loop):
	lvx	v4, 0, rSTR1
	lvx	v5, 0, rSTR2
	vcmpequb.	v7, v6, v8
	bnl	cr6, L(different3)
	lvx	v6, rSTR1, r8
	lvx	v8, rSTR2, r8
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different2)
	lvx	v4, rSTR1, r10
	lvx	v5, rSTR2, r10
	vcmpequb.	v7, v6, v8
	bnl	cr6, L(different3)
	lvx	v6, rSTR1, r11
	lvx	v8, rSTR2, r11
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different2)
	addi	rSTR1, rSTR1, 64
	addi	rSTR2, rSTR2, 64
	bdnz	L(aligned_loop)
	vcmpequb.	v7, v6, v8
	bnl	cr6, L(different3)
	clrldi	rN, rN, 58
	/* Handle remainder for aligned loop.  */
	.align	4
L(lessthan64):
	mr	r9, rSTR1
	cmpdi	cr6, rN, 0
	li	rSTR1, 0
	blelr	cr6
	lvx	v4, 0, r9
	lvx	v5, 0, rSTR2
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	addi	rN, rN, -16

	cmpdi	cr6, rN, 0
	blelr	cr6
	lvx	v4, r9, r8
	lvx	v5, rSTR2, r8
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	addi	rN, rN, -16

	cmpdi	cr6, rN, 0
	blelr	cr6
	lvx	v4, r9, r10
	lvx	v5, rSTR2, r10
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	addi	rN, rN, -16

	cmpdi	cr6, rN, 0
	blelr	cr6
	lvx	v4, r9, r11
	lvx	v5, rSTR2, r11
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	blr

	/* Calculate and return the difference.  */
	.align 4
L(different1):
	cmpdi	cr6, rN, 16
	bge	cr6, L(different2)
	/* Discard unwanted bytes.  */
#ifdef __LITTLE_ENDIAN__
	lvsr	v1, 0, rN
	vperm	v4, v4, v0, v1
	vperm	v5, v5, v0, v1
#else
	lvsl	v1, 0, rN
	vperm	v4, v0, v4, v1
	vperm	v5, v0, v5, v1
#endif
	vcmpequb.	v7, v4, v5
	li	rRTN, 0
	bltlr	cr6
	.align 4
L(different2):
#ifdef __LITTLE_ENDIAN__
	/* Reverse bytes for direct comparison.  */
	lvsl	v10, r0, r0
	vspltisb	v8, 15
	vsububm	v9, v8, v10
	vperm	v4, v4, v0, v9
	vperm	v5, v5, v0, v9
#endif
	mfvrd	r7, v4
	mfvrd	r9, v5
	cmpld	cr6, r7, r9
	bne	cr6, L(ret_diff)
	/* Difference in second DW.  */
	vsldoi	v4, v4, v4, 8
	vsldoi	v5, v5, v5, 8
	mfvrd	r7, v4
	mfvrd	r9, v5
	cmpld	cr6, r7, r9
L(ret_diff):
	li	rRTN, 1
	bgtlr	cr6
	li	rRTN, -1
	blr
	.align	4
L(different3):
#ifdef __LITTLE_ENDIAN__
	/* Reverse bytes for direct comparison.  */
	vspltisb	v9, 15
	lvsl	v10, r0, r0
	vsububm	v9, v9, v10
	vperm	v6, v6, v0, v9
	vperm	v8, v8, v0, v9
#endif
	mfvrd	r7, v6
	mfvrd	r9, v8
	cmpld	cr6, r7, r9
	bne	cr6, L(ret_diff)
	/* Difference in second DW.  */
	vsldoi	v6, v6, v6, 8
	vsldoi	v8, v8, v8, 8
	mfvrd	r7, v6
	mfvrd	r9, v8
	cmpld	cr6, r7, r9
	li	rRTN, 1
	bgtlr	cr6
	li	rRTN, -1
	blr

	.align 4
L(different):
	cmpldi	cr7, rN, 8
	bgt	cr7, L(end)
	/* Skip unwanted bytes.  */
	sldi	r8, rN, 3
	subfic	r8, r8, 64
	srd	rWORD1, rWORD1, r8
	srd	rWORD2, rWORD2, r8
	cmpld	cr6, rWORD1, rWORD2
	li	rRTN, 0
	beqlr	cr6
L(end):
	li	rRTN, 1
	bgtlr	cr6
	li	rRTN, -1
	blr

	.align	4
L(unalignedqw):
	/* Proceed to DW unaligned loop,if there is a chance of pagecross.  */
	rldicl	r9, rSTR1, 0, 52
	add	r9, r9, rN
	cmpldi	cr0, r9, 4096-16
	bgt	cr0, L(unaligned)
	rldicl	r9, rSTR2, 0, 52
	add	r9, r9, rN
	cmpldi	cr0, r9, 4096-16
	bgt	cr0, L(unaligned)
	li	r0, 0
	li	r8, 16
	vspltisb	v0, 0
	/* Check if rSTR1 is aligned to QW.  */
	andi.	r11, rSTR1, 0xF
	beq	L(s1_align)

	/* Compare 16B and align S1 to QW.  */
#ifdef __LITTLE_ENDIAN__
	lvsr	v10, 0, rSTR1	/* Compute mask.  */
	lvsr	v6, 0, rSTR2	/* Compute mask.  */
#else
	lvsl	v10, 0, rSTR1	/* Compute mask.  */
	lvsl	v6, 0, rSTR2	/* Compute mask.  */
#endif
	lvx	v5, 0, rSTR2
	lvx	v9, rSTR2, r8
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v9, v5, v6
#else
	vperm	v5, v5, v9, v6
#endif
	lvx	v4, 0, rSTR1
	lvx	v9, rSTR1, r8
#ifdef __LITTLE_ENDIAN__
	vperm	v4, v9, v4, v10
#else
	vperm	v4, v4, v9, v10
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	cmpldi	cr6, rN, 16
	ble	cr6, L(zeroLength)
	subfic	r11, r11, 16
	subf	rN, r11, rN
	add	rSTR1, rSTR1, r11
	add	rSTR2, rSTR2, r11

	/* As s1 is QW aligned prepare for unaligned loop.  */
	.align	4
L(s1_align):
#ifdef __LITTLE_ENDIAN__
	lvsr	v6, 0, rSTR2
#else
	lvsl	v6, 0, rSTR2
#endif
	lvx	v5, 0, rSTR2
	srdi.	r6, rN, 6
	li	r10, 32
	li	r11, 48
	ble	cr0, L(lessthan64_unalign)
	mtctr	r6
	li 	r9, 64
	/* Unaligned vector loop.  */
	.align	4
L(unalign_qwloop):
	lvx	v4, 0, rSTR1
	lvx	v10, rSTR2, r8
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different2)
	vor	v5, v10, v10
	lvx	v4, rSTR1, r8
	lvx	v10, rSTR2, r10
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different2)
	vor	v5, v10, v10
	lvx	v4, rSTR1, r10
	lvx	v10, rSTR2, r11
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different2)
	vor	v5, v10, v10
	lvx	v4, rSTR1, r11
	lvx	v10, rSTR2, r9
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different2)
	vor	v5, v10, v10
	addi	rSTR1, rSTR1, 64
	addi	rSTR2, rSTR2, 64
	bdnz	L(unalign_qwloop)
	clrldi	rN, rN, 58
	/* Handle remainder for unaligned loop.  */
	.align	4
L(lessthan64_unalign):
	mr	r9, rSTR1
	cmpdi	cr6, rN, 0
	li	rSTR1, 0
	blelr	cr6
	lvx	v4, 0, r9
	lvx     v10, rSTR2, r8
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	vor	v5, v10, v10
	addi	rN, rN, -16

	cmpdi	cr6, rN, 0
	blelr	cr6
	lvx	v4, r9, r8
	lvx	v10, rSTR2, r10
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	vor	v5, v10, v10
	addi	rN, rN, -16

	cmpdi	cr6, rN, 0
	blelr	cr6
	lvx	v4, r9, r10
	lvx	v10, rSTR2, r11
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	vor	v5, v10, v10
	addi	rN, rN, -16

	cmpdi	cr6, rN, 0
	blelr	cr6
	lvx	v4, r9, r11
	addi	r11, r11, 16
	lvx	v10, rSTR2, r11
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v10, v5, v6
#else
	vperm	v5, v5, v10, v6
#endif
	vcmpequb.	v7, v5, v4
	bnl	cr6, L(different1)
	blr

/* Otherwise we know the two strings have the same alignment (but not
   yet DW).  So we force the string addresses to the next lower DW
   boundary and special case this first DW using shift left to
   eliminate bits preceding the first byte.  Since we want to join the
   normal (DW aligned) compare loop, starting at the second double word,
   we need to adjust the length (rN) and special case the loop
   versioning for the first DW.  This ensures that the loop count is
   correct and the first DW (shifted) is in the expected register pair.  */
	.align	4
L(DW):
	std	rWORD8, rWORD8SAVE(r1)
	std	rWORD7, rWORD7SAVE(r1)
	std	rOFF8, rOFF8SAVE(r1)
	std	rOFF16, rOFF16SAVE(r1)
	std	rOFF24, rOFF24SAVE(r1)
	std	rOFF32, rOFF32SAVE(r1)
	cfi_offset(rWORD8, rWORD8SAVE)
	cfi_offset(rWORD7, rWORD7SAVE)
	cfi_offset(rOFF8, rOFF8SAVE)
	cfi_offset(rOFF16, rOFF16SAVE)
	cfi_offset(rOFF24, rOFF24SAVE)
	cfi_offset(rOFF32, rOFF32SAVE)

	li	rOFF8,8
	li	rOFF16,16
	li	rOFF24,24
	li	rOFF32,32
	clrrdi	rSTR1, rSTR1, 3
	clrrdi	rSTR2, rSTR2, 3
	beq	cr5, L(DWaligned)
	add	rN, rN, r12
	sldi	rWORD6, r12, 3
	srdi	r0, rN, 5	/* Divide by 32.  */
	andi.	r12, rN, 24	/* Get the DW remainder.  */
	LD	rWORD1, 0, rSTR1
	LD	rWORD2, 0, rSTR2
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	clrldi	rN, rN, 61
	beq	L(dPs4)
	mtctr	r0
	bgt	cr1, L(dPs3)
	beq	cr1, L(dPs2)

/* Remainder is 8.  */
	.align	3
L(dsP1):
	sld	rWORD5, rWORD1, rWORD6
	sld	rWORD6, rWORD2, rWORD6
	cmpld	cr5, rWORD5, rWORD6
	blt	cr7, L(dP1x)
/* Do something useful in this cycle since we have to branch anyway.  */
	LD	rWORD1, rOFF8, rSTR1
	LD	rWORD2, rOFF8, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	b	L(dP1e)
/* Remainder is 16.  */
	.align	4
L(dPs2):
	sld	rWORD5, rWORD1, rWORD6
	sld	rWORD6, rWORD2, rWORD6
	cmpld	cr6, rWORD5, rWORD6
	blt	cr7, L(dP2x)
/* Do something useful in this cycle since we have to branch anyway.  */
	LD	rWORD7, rOFF8, rSTR1
	LD	rWORD8, rOFF8, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	b	L(dP2e)
/* Remainder is 24.  */
	.align	4
L(dPs3):
	sld	rWORD3, rWORD1, rWORD6
	sld	rWORD4, rWORD2, rWORD6
	cmpld	cr1, rWORD3, rWORD4
	b	L(dP3e)
/* Count is a multiple of 32, remainder is 0.  */
	.align	4
L(dPs4):
	mtctr	r0
	sld	rWORD1, rWORD1, rWORD6
	sld	rWORD2, rWORD2, rWORD6
	cmpld	cr7, rWORD1, rWORD2
	b	L(dP4e)

/* At this point we know both strings are double word aligned and the
   compare length is at least 8 bytes.  */
	.align	4
L(DWaligned):
	andi.	r12, rN, 24	/* Get the DW remainder.  */
	srdi	r0, rN, 5	/* Divide by 32.  */
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	clrldi	rN, rN, 61
	beq	L(dP4)
	bgt	cr1, L(dP3)
	beq	cr1, L(dP2)

/* Remainder is 8.  */
	.align	4
L(dP1):
	mtctr	r0
/* Normally we'd use rWORD7/rWORD8 here, but since we might exit early
   (8-15 byte compare), we want to use only volatile registers.  This
   means we can avoid restoring non-volatile registers since we did not
   change any on the early exit path.  The key here is the non-early
   exit path only cares about the condition code (cr5), not about which
   register pair was used.  */
	LD	rWORD5, 0, rSTR1
	LD	rWORD6, 0, rSTR2
	cmpld	cr5, rWORD5, rWORD6
	blt	cr7, L(dP1x)
	LD	rWORD1, rOFF8, rSTR1
	LD	rWORD2, rOFF8, rSTR2
	cmpld	cr7, rWORD1, rWORD2
L(dP1e):
	LD	rWORD3, rOFF16, rSTR1
	LD	rWORD4, rOFF16, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	LD	rWORD5, rOFF24, rSTR1
	LD	rWORD6, rOFF24, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(dLcr5x)
	bne	cr7, L(dLcr7x)

	LD	rWORD7, rOFF32, rSTR1
	LD	rWORD8, rOFF32, rSTR2
	addi	rSTR1, rSTR1, 32
	addi	rSTR2, rSTR2, 32
	bne	cr1, L(dLcr1)
	cmpld	cr5, rWORD7, rWORD8
	bdnz	L(dLoop)
	bne	cr6, L(dLcr6)
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
	.align	3
L(dP1x):
	sldi.	r12, rN, 3
	bne	cr5, L(dLcr5x)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	bne	L(d00)
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 0
	blr

/* Remainder is 16.  */
	.align	4
L(dP2):
	mtctr	r0
	LD	rWORD5, 0, rSTR1
	LD	rWORD6, 0, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	blt	cr7, L(dP2x)
	LD	rWORD7, rOFF8, rSTR1
	LD	rWORD8, rOFF8, rSTR2
	cmpld	cr5, rWORD7, rWORD8
L(dP2e):
	LD	rWORD1, rOFF16, rSTR1
	LD	rWORD2, rOFF16, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	LD	rWORD3, rOFF24, rSTR1
	LD	rWORD4, rOFF24, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
	bne	cr6, L(dLcr6)
	bne	cr5, L(dLcr5)
	b	L(dLoop2)
	.align	4
L(dP2x):
	LD	rWORD3, rOFF8, rSTR1
	LD	rWORD4, rOFF8, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	sldi.	r12, rN, 3
	bne	cr6, L(dLcr6x)
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
	bne	cr1, L(dLcr1x)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	bne	L(d00)
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 0
	blr

/* Remainder is 24.  */
	.align	4
L(dP3):
	mtctr	r0
	LD	rWORD3, 0, rSTR1
	LD	rWORD4, 0, rSTR2
	cmpld	cr1, rWORD3, rWORD4
L(dP3e):
	LD	rWORD5, rOFF8, rSTR1
	LD	rWORD6, rOFF8, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	blt	cr7, L(dP3x)
	LD	rWORD7, rOFF16, rSTR1
	LD	rWORD8, rOFF16, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	LD	rWORD1, rOFF24, rSTR1
	LD	rWORD2, rOFF24, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
	bne	cr1, L(dLcr1)
	bne	cr6, L(dLcr6)
	b	L(dLoop1)
/* Again we are on a early exit path (24-31 byte compare), we want to
   only use volatile registers and avoid restoring non-volatile
   registers.  */
	.align	4
L(dP3x):
	LD	rWORD1, rOFF16, rSTR1
	LD	rWORD2, rOFF16, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	sldi.	r12, rN, 3
	bne	cr1, L(dLcr1x)
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
	bne	cr6, L(dLcr6x)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	bne	cr7, L(dLcr7x)
	bne	L(d00)
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 0
	blr

/* Count is a multiple of 32, remainder is 0.  */
	.align	4
L(dP4):
	mtctr	r0
	LD	rWORD1, 0, rSTR1
	LD	rWORD2, 0, rSTR2
	cmpld	cr7, rWORD1, rWORD2
L(dP4e):
	LD	rWORD3, rOFF8, rSTR1
	LD	rWORD4, rOFF8, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	LD	rWORD5, rOFF16, rSTR1
	LD	rWORD6, rOFF16, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	LD	rWORD7, rOFF24, rSTR1
	LD	rWORD8, rOFF24, rSTR2
	addi	rSTR1, rSTR1, 24
	addi	rSTR2, rSTR2, 24
	cmpld	cr5, rWORD7, rWORD8
	bne	cr7, L(dLcr7)
	bne	cr1, L(dLcr1)
	bdz-	L(d24)		/* Adjust CTR as we start with +4.  */
/* This is the primary loop.  */
	.align	4
L(dLoop):
	LD	rWORD1, rOFF8, rSTR1
	LD	rWORD2, rOFF8, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	bne	cr6, L(dLcr6)
L(dLoop1):
	LD	rWORD3, rOFF16, rSTR1
	LD	rWORD4, rOFF16, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(dLcr5)
L(dLoop2):
	LD	rWORD5, rOFF24, rSTR1
	LD	rWORD6, rOFF24, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	bne	cr7, L(dLcr7)
L(dLoop3):
	LD	rWORD7, rOFF32, rSTR1
	LD	rWORD8, rOFF32, rSTR2
	addi	rSTR1, rSTR1, 32
	addi	rSTR2, rSTR2, 32
	bne	cr1, L(dLcr1)
	cmpld	cr7, rWORD1, rWORD2
	bdnz	L(dLoop)

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
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
	subfic	rN, r12, 64	/* Shift count is 64 - (rN * 8).  */
	beq	L(duzeroLength)
/* At this point we have a remainder of 1 to 7 bytes to compare.  Since
   we are aligned it is safe to load the whole double word, and use
   shift right double to eliminate bits beyond the compare length.  */
L(d00):
	LD	rWORD1, rOFF8, rSTR1
	LD	rWORD2, rOFF8, rSTR2
	srd	rWORD1, rWORD1, rN
	srd	rWORD2, rWORD2, rN
	cmpld	cr7, rWORD1, rWORD2
	bne	cr7, L(dLcr7x)
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 0
	blr

	.align	4
L(dLcr7):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
L(dLcr7x):
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 1
	bgtlr	cr7
	li	rRTN, -1
	blr
	.align	4
L(dLcr1):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
L(dLcr1x):
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 1
	bgtlr	cr1
	li	rRTN, -1
	blr
	.align	4
L(dLcr6):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
L(dLcr6x):
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 1
	bgtlr	cr6
	li	rRTN, -1
	blr
	.align	4
L(dLcr5):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
L(dLcr5x):
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 1
	bgtlr	cr5
	li	rRTN, -1
	blr

	.align	4
L(bytealigned):
	mtctr	rN

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
	bdz	L(b11)
	cmpld	cr7, rWORD1, rWORD2
	lbz	rWORD3, 1(rSTR1)
	lbz	rWORD4, 1(rSTR2)
	bdz	L(b12)
	cmpld	cr1, rWORD3, rWORD4
	lbzu	rWORD5, 2(rSTR1)
	lbzu	rWORD6, 2(rSTR2)
	bdz	L(b13)
	.align	4
L(bLoop):
	lbzu	rWORD1, 1(rSTR1)
	lbzu	rWORD2, 1(rSTR2)
	bne	cr7, L(bLcr7)

	cmpld	cr6, rWORD5, rWORD6
	bdz	L(b3i)

	lbzu	rWORD3, 1(rSTR1)
	lbzu	rWORD4, 1(rSTR2)
	bne	cr1, L(bLcr1)

	cmpld	cr7, rWORD1, rWORD2
	bdz	L(b2i)

	lbzu	rWORD5, 1(rSTR1)
	lbzu	rWORD6, 1(rSTR2)
	bne	cr6, L(bLcr6)

	cmpld	cr1, rWORD3, rWORD4
	bdnz	L(bLoop)

/* We speculatively loading bytes before we have tested the previous
   bytes.  But we must avoid overrunning the length (in the ctr) to
   prevent these speculative loads from causing a segfault.  In this
   case the loop will exit early (before the all pending bytes are
   tested.  In this case we must complete the pending operations
   before returning.  */
L(b1i):
	bne	cr7, L(bLcr7)
	bne	cr1, L(bLcr1)
	b	L(bx56)
	.align	4
L(b2i):
	bne	cr6, L(bLcr6)
	bne	cr7, L(bLcr7)
	b	L(bx34)
	.align	4
L(b3i):
	bne	cr1, L(bLcr1)
	bne	cr6, L(bLcr6)
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
	bne	cr7, L(bx12)
	bne	cr1, L(bx34)
L(bx56):
	sub	rRTN, rWORD5, rWORD6
	blr
	nop
L(b12):
	bne	cr7, L(bx12)
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
   versioning for the first DW.  This ensures that the loop count is
   correct and the first DW (shifted) is in the expected resister pair.  */
L(unaligned):
	std	rWORD8, rWORD8SAVE(r1)
	std	rWORD7, rWORD7SAVE(r1)
	std	rOFF8, rOFF8SAVE(r1)
	std	rOFF16, rOFF16SAVE(r1)
	std	rOFF24, rOFF24SAVE(r1)
	std	rOFF32, rOFF32SAVE(r1)
	cfi_offset(rWORD8, rWORD8SAVE)
	cfi_offset(rWORD7, rWORD7SAVE)
	cfi_offset(rOFF8, rOFF8SAVE)
	cfi_offset(rOFF16, rOFF16SAVE)
	cfi_offset(rOFF24, rOFF24SAVE)
	cfi_offset(rOFF32, rOFF32SAVE)
	li	rOFF8,8
	li	rOFF16,16
	li	rOFF24,24
	li	rOFF32,32
	std	rSHL, rSHLSAVE(r1)
	cfi_offset(rSHL, rSHLSAVE)
	clrldi	rSHL, rSTR2, 61
	beq	cr6, L(duzeroLength)
	std	rSHR, rSHRSAVE(r1)
	cfi_offset(rSHR, rSHRSAVE)
	beq	cr5, L(DWunaligned)
	std	rWORD8_SHIFT, rWORD8SHIFTSAVE(r1)
	cfi_offset(rWORD8_SHIFT, rWORD8SHIFTSAVE)
/* Adjust the logical start of rSTR2 to compensate for the extra bits
   in the 1st rSTR1 DW.  */
	sub	rWORD8_SHIFT, rSTR2, r12
/* But do not attempt to address the DW before that DW that contains
   the actual start of rSTR2.  */
	clrrdi	rSTR2, rSTR2, 3
	std	rWORD2_SHIFT, rWORD2SHIFTSAVE(r1)
/* Compute the left/right shift counts for the unaligned rSTR2,
   compensating for the logical (DW aligned) start of rSTR1.  */
	clrldi	rSHL, rWORD8_SHIFT, 61
	clrrdi	rSTR1, rSTR1, 3
	std	rWORD4_SHIFT, rWORD4SHIFTSAVE(r1)
	sldi	rSHL, rSHL, 3
	cmpld	cr5, rWORD8_SHIFT, rSTR2
	add	rN, rN, r12
	sldi	rWORD6, r12, 3
	std	rWORD6_SHIFT, rWORD6SHIFTSAVE(r1)
	cfi_offset(rWORD2_SHIFT, rWORD2SHIFTSAVE)
	cfi_offset(rWORD4_SHIFT, rWORD4SHIFTSAVE)
	cfi_offset(rWORD6_SHIFT, rWORD6SHIFTSAVE)
	subfic	rSHR, rSHL, 64
	srdi	r0, rN, 5	/* Divide by 32.  */
	andi.	r12, rN, 24	/* Get the DW remainder.  */
/* We normally need to load 2 DWs to start the unaligned rSTR2, but in
   this special case those bits may be discarded anyway.  Also we
   must avoid loading a DW where none of the bits are part of rSTR2 as
   this may cross a page boundary and cause a page fault.  */
	li	rWORD8, 0
	blt	cr5, L(dus0)
	LD	rWORD8, 0, rSTR2
	addi	rSTR2, rSTR2, 8
	sld	rWORD8, rWORD8, rSHL

L(dus0):
	LD	rWORD1, 0, rSTR1
	LD	rWORD2, 0, rSTR2
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	srd	r12, rWORD2, rSHR
	clrldi	rN, rN, 61
	beq	L(duPs4)
	mtctr	r0
	or	rWORD8, r12, rWORD8
	bgt	cr1, L(duPs3)
	beq	cr1, L(duPs2)

/* Remainder is 8.  */
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
	LD	rWORD2, rOFF8, rSTR2
	srd	r0, rWORD2, rSHR
	b	L(dutrim)
/* Remainder is 16.  */
	.align	4
L(duPs2):
	sld	rWORD6_SHIFT, rWORD2, rSHL
	sld	rWORD5, rWORD1, rWORD6
	sld	rWORD6, rWORD8, rWORD6
	b	L(duP2e)
/* Remainder is 24.  */
	.align	4
L(duPs3):
	sld	rWORD4_SHIFT, rWORD2, rSHL
	sld	rWORD3, rWORD1, rWORD6
	sld	rWORD4, rWORD8, rWORD6
	b	L(duP3e)
/* Count is a multiple of 32, remainder is 0.  */
	.align	4
L(duPs4):
	mtctr	r0
	or	rWORD8, r12, rWORD8
	sld	rWORD2_SHIFT, rWORD2, rSHL
	sld	rWORD1, rWORD1, rWORD6
	sld	rWORD2, rWORD8, rWORD6
	b	L(duP4e)

/* At this point we know rSTR1 is double word aligned and the
   compare length is at least 8 bytes.  */
	.align	4
L(DWunaligned):
	std	rWORD8_SHIFT, rWORD8SHIFTSAVE(r1)
	clrrdi	rSTR2, rSTR2, 3
	std	rWORD2_SHIFT, rWORD2SHIFTSAVE(r1)
	srdi	r0, rN, 5	/* Divide by 32.  */
	std	rWORD4_SHIFT, rWORD4SHIFTSAVE(r1)
	andi.	r12, rN, 24	/* Get the DW remainder.  */
	std	rWORD6_SHIFT, rWORD6SHIFTSAVE(r1)
	cfi_offset(rWORD8_SHIFT, rWORD8SHIFTSAVE)
	cfi_offset(rWORD2_SHIFT, rWORD2SHIFTSAVE)
	cfi_offset(rWORD4_SHIFT, rWORD4SHIFTSAVE)
	cfi_offset(rWORD6_SHIFT, rWORD6SHIFTSAVE)
	sldi	rSHL, rSHL, 3
	LD	rWORD6, 0, rSTR2
	LD	rWORD8, rOFF8, rSTR2
	addi	rSTR2, rSTR2, 8
	cmpldi	cr1, r12, 16
	cmpldi	cr7, rN, 32
	clrldi	rN, rN, 61
	subfic	rSHR, rSHL, 64
	sld	rWORD6_SHIFT, rWORD6, rSHL
	beq	L(duP4)
	mtctr	r0
	bgt	cr1, L(duP3)
	beq	cr1, L(duP2)

/* Remainder is 8.  */
	.align	4
L(duP1):
	srd	r12, rWORD8, rSHR
	LD	rWORD7, 0, rSTR1
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	blt	cr7, L(duP1x)
L(duP1e):
	LD	rWORD1, rOFF8, rSTR1
	LD	rWORD2, rOFF8, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
	LD	rWORD3, rOFF16, rSTR1
	LD	rWORD4, rOFF16, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	bne	cr5, L(duLcr5)
	or	rWORD4, r12, rWORD2_SHIFT
	LD	rWORD5, rOFF24, rSTR1
	LD	rWORD6, rOFF24, rSTR2
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
	LD	rWORD2, rOFF8, rSTR2
	srd	r0, rWORD2, rSHR
	b	L(dutrim)
/* Remainder is 16.  */
	.align	4
L(duP2):
	srd	r0, rWORD8, rSHR
	LD	rWORD5, 0, rSTR1
	or	rWORD6, r0, rWORD6_SHIFT
	sld	rWORD6_SHIFT, rWORD8, rSHL
L(duP2e):
	LD	rWORD7, rOFF8, rSTR1
	LD	rWORD8, rOFF8, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	blt	cr7, L(duP2x)
	LD	rWORD1, rOFF16, rSTR1
	LD	rWORD2, rOFF16, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	bne	cr6, L(duLcr6)
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
	LD	rWORD3, rOFF24, rSTR1
	LD	rWORD4, rOFF24, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	bne	cr5, L(duLcr5)
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	or	rWORD4, r12, rWORD2_SHIFT
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
	cmpld	cr1, rWORD3, rWORD4
	b	L(duLoop2)
	.align	4
L(duP2x):
	cmpld	cr5, rWORD7, rWORD8
	addi	rSTR1, rSTR1, 8
	addi	rSTR2, rSTR2, 8
	bne	cr6, L(duLcr6)
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
	LD	rWORD2, rOFF8, rSTR2
	srd	r0, rWORD2, rSHR
	b	L(dutrim)

/* Remainder is 24.  */
	.align	4
L(duP3):
	srd	r12, rWORD8, rSHR
	LD	rWORD3, 0, rSTR1
	sld	rWORD4_SHIFT, rWORD8, rSHL
	or	rWORD4, r12, rWORD6_SHIFT
L(duP3e):
	LD	rWORD5, rOFF8, rSTR1
	LD	rWORD6, rOFF8, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	or	rWORD6, r0, rWORD4_SHIFT
	LD	rWORD7, rOFF16, rSTR1
	LD	rWORD8, rOFF16, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	bne	cr1, L(duLcr1)
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	blt	cr7, L(duP3x)
	LD	rWORD1, rOFF24, rSTR1
	LD	rWORD2, rOFF24, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	bne	cr6, L(duLcr6)
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
	cmpld	cr7, rWORD1, rWORD2
	b	L(duLoop1)
	.align	4
L(duP3x):
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
	cmpld	cr5, rWORD7, rWORD8
	bne	cr6, L(duLcr6)
	sldi.	rN, rN, 3
	bne	cr5, L(duLcr5)
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
	LD	rWORD2, rOFF8, rSTR2
	srd	r0, rWORD2, rSHR
	b	L(dutrim)

/* Count is a multiple of 32, remainder is 0.  */
	.align	4
L(duP4):
	mtctr	r0
	srd	r0, rWORD8, rSHR
	LD	rWORD1, 0, rSTR1
	sld	rWORD2_SHIFT, rWORD8, rSHL
	or	rWORD2, r0, rWORD6_SHIFT
L(duP4e):
	LD	rWORD3, rOFF8, rSTR1
	LD	rWORD4, rOFF8, rSTR2
	cmpld	cr7, rWORD1, rWORD2
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	or	rWORD4, r12, rWORD2_SHIFT
	LD	rWORD5, rOFF16, rSTR1
	LD	rWORD6, rOFF16, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	bne	cr7, L(duLcr7)
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	or	rWORD6, r0, rWORD4_SHIFT
	LD	rWORD7, rOFF24, rSTR1
	LD	rWORD8, rOFF24, rSTR2
	addi	rSTR1, rSTR1, 24
	addi	rSTR2, rSTR2, 24
	cmpld	cr6, rWORD5, rWORD6
	bne	cr1, L(duLcr1)
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	cmpld	cr5, rWORD7, rWORD8
	bdz	L(du24)		/* Adjust CTR as we start with +4.  */
/* This is the primary loop.  */
	.align	4
L(duLoop):
	LD	rWORD1, rOFF8, rSTR1
	LD	rWORD2, rOFF8, rSTR2
	cmpld	cr1, rWORD3, rWORD4
	bne	cr6, L(duLcr6)
	srd	r0, rWORD2, rSHR
	sld	rWORD2_SHIFT, rWORD2, rSHL
	or	rWORD2, r0, rWORD8_SHIFT
L(duLoop1):
	LD	rWORD3, rOFF16, rSTR1
	LD	rWORD4, rOFF16, rSTR2
	cmpld	cr6, rWORD5, rWORD6
	bne	cr5, L(duLcr5)
	srd	r12, rWORD4, rSHR
	sld	rWORD4_SHIFT, rWORD4, rSHL
	or	rWORD4, r12, rWORD2_SHIFT
L(duLoop2):
	LD	rWORD5, rOFF24, rSTR1
	LD	rWORD6, rOFF24, rSTR2
	cmpld	cr5, rWORD7, rWORD8
	bne	cr7, L(duLcr7)
	srd	r0, rWORD6, rSHR
	sld	rWORD6_SHIFT, rWORD6, rSHL
	or	rWORD6, r0, rWORD4_SHIFT
L(duLoop3):
	LD	rWORD7, rOFF32, rSTR1
	LD	rWORD8, rOFF32, rSTR2
	addi	rSTR1, rSTR1, 32
	addi	rSTR2, rSTR2, 32
	cmpld	cr7, rWORD1, rWORD2
	bne	cr1, L(duLcr1)
	srd	r12, rWORD8, rSHR
	sld	rWORD8_SHIFT, rWORD8, rSHL
	or	rWORD8, r12, rWORD6_SHIFT
	bdnz	L(duLoop)

L(duL4):
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
   string length.  So we compare the bit length of the remainder to
   the right shift count (rSHR).  If the bit count is less than or equal
   we do not need to load rWORD2 (all significant bits are already in
   rWORD8_SHIFT).  */
	cmpld	cr7, rN, rSHR
	beq	L(duZeroReturn)
	li	r0, 0
	ble	cr7, L(dutrim)
	LD	rWORD2, rOFF8, rSTR2
	srd	r0, rWORD2, rSHR
	.align	4
L(dutrim):
	LD	rWORD1, rOFF8, rSTR1
	ld	rWORD8, -8(r1)
	subfic	rN, rN, 64	/* Shift count is 64 - (rN * 8).  */
	or	rWORD2, r0, rWORD8_SHIFT
	ld	rWORD7, rWORD7SAVE(r1)
	ld	rSHL, rSHLSAVE(r1)
	srd	rWORD1, rWORD1, rN
	srd	rWORD2, rWORD2, rN
	ld	rSHR, rSHRSAVE(r1)
	ld	rWORD8_SHIFT, rWORD8SHIFTSAVE(r1)
	li	rRTN, 0
	cmpld	cr7, rWORD1, rWORD2
	ld	rWORD2_SHIFT, rWORD2SHIFTSAVE(r1)
	ld	rWORD4_SHIFT, rWORD4SHIFTSAVE(r1)
	beq	cr7, L(dureturn24)
	li	rRTN, 1
	ld	rWORD6_SHIFT, rWORD6SHIFTSAVE(r1)
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	bgtlr	cr7
	li	rRTN, -1
	blr
	.align	4
L(duLcr7):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
	li	rRTN, 1
	bgt	cr7, L(dureturn29)
	ld	rSHL, rSHLSAVE(r1)
	ld	rSHR, rSHRSAVE(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	4
L(duLcr1):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
	li	rRTN, 1
	bgt	cr1, L(dureturn29)
	ld	rSHL, rSHLSAVE(r1)
	ld	rSHR, rSHRSAVE(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	4
L(duLcr6):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
	li	rRTN, 1
	bgt	cr6, L(dureturn29)
	ld	rSHL, rSHLSAVE(r1)
	ld	rSHR, rSHRSAVE(r1)
	li	rRTN, -1
	b	L(dureturn27)
	.align	4
L(duLcr5):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
	li	rRTN, 1
	bgt	cr5, L(dureturn29)
	ld	rSHL, rSHLSAVE(r1)
	ld	rSHR, rSHRSAVE(r1)
	li	rRTN, -1
	b	L(dureturn27)

	.align	3
L(duZeroReturn):
	li	rRTN, 0
	.align	4
L(dureturn):
	ld	rWORD8, rWORD8SAVE(r1)
	ld	rWORD7, rWORD7SAVE(r1)
L(dureturn29):
	ld	rSHL, rSHLSAVE(r1)
	ld	rSHR, rSHRSAVE(r1)
L(dureturn27):
	ld	rWORD8_SHIFT, rWORD8SHIFTSAVE(r1)
	ld	rWORD2_SHIFT, rWORD2SHIFTSAVE(r1)
	ld	rWORD4_SHIFT, rWORD4SHIFTSAVE(r1)
L(dureturn24):
	ld	rWORD6_SHIFT, rWORD6SHIFTSAVE(r1)
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	blr

L(duzeroLength):
	ld	rOFF8,  rOFF8SAVE(r1)
	ld	rOFF16, rOFF16SAVE(r1)
	ld	rOFF24, rOFF24SAVE(r1)
	ld	rOFF32, rOFF32SAVE(r1)
	li	rRTN, 0
	blr

END (MEMCMP)
libc_hidden_builtin_def (memcmp)
weak_alias (memcmp, bcmp)
