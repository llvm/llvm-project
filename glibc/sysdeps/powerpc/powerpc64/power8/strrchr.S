/* Optimized strrchr implementation for PowerPC64/POWER7 using cmpb insn.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* char *[r3] strrchr (char *s [r3], int c [r4])  */

#ifdef __LITTLE_ENDIAN__
/* Find the match position from v6 and place result in r6.  */
# define CALCULATE_MATCH() \
	vbpermq	v6, v6, v10; \
	vsldoi	v6, v6, v6, 6; \
	mfvrd	r7, v6; \
	cntlzd	r6, r7; \
	subfic	r6, r6, 15;
/*
 * Find the first null position to mask bytes after null.
 * (reg): vcmpequb result: v2 for 1st qw v3 for 2nd qw.
 * Result placed at v2.
 */
# define FIND_NULL_POS(reg) \
	vspltisb	v11, -1; \
	vadduqm	v11, reg, v11; \
	vandc	v11, v11, reg; \
	vpopcntd	v2, v11; \
	vspltb	v11, v2, 15; \
	vcmpequb.	v11, v11, v9; \
	blt	cr6, 1f; \
	vsldoi	v9, v0, v9, 1; \
	vslo	v2, v2, v9; \
1: \
	vsumsws	v2, v2, v0;
#else
# define CALCULATE_MATCH() \
	vbpermq	v6, v6, v10; \
	mfvrd	r7, v6; \
	addi	r6, r7, -1; \
	andc	r6, r6, r7; \
	popcntd	r6, r6; \
	subfic	r6, r6, 15;
# define FIND_NULL_POS(reg) \
	vclzd	v2, reg; \
	vspltb	v11, v2, 7; \
	vcmpequb.	v11, v11, v9; \
	blt	cr6, 1f; \
	vsldoi	v9, v0, v9, 1; \
	vsro	v2, v2, v9; \
1: \
	vsumsws	v2, v2, v0;
#endif	/* !__LITTLE_ENDIAN__  */

#ifndef STRRCHR
# define STRRCHR strrchr
#endif
	.machine  power8
ENTRY_TOCLESS (STRRCHR)
	CALL_MCOUNT 2
	dcbt	0,r3
	clrrdi	r8,r3,3	      /* Align the address to doubleword boundary.  */
	cmpdi	cr7,r4,0
	ld	r12,0(r8)     /* Load doubleword from memory.  */
	li	r9,0	      /* Used to store last occurence.  */
	li	r0,0	      /* Doubleword with null chars to use
				 with cmpb.  */

	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */

	beq	cr7,L(null_match)

	/* Replicate byte to doubleword.  */
	insrdi	r4,r4,8,48
	insrdi	r4,r4,16,32
	insrdi	r4,r4,32,0

	/* r4 is changed now.  If it's passed more chars, then
	   check for null again.  */
	cmpdi	cr7,r4,0
	beq	cr7,L(null_match)
	/* Now r4 has a doubleword of c bytes and r0 has
	   a doubleword of null bytes.  */

	cmpb	r10,r12,r4     /* Compare each byte against c byte.  */
	cmpb	r11,r12,r0     /* Compare each byte against null byte.  */

	/* Move the doublewords left and right to discard the bits that are
	   not part of the string and bring them back as zeros.  */
#ifdef __LITTLE_ENDIAN__
	srd	r10,r10,r6
	srd	r11,r11,r6
	sld	r10,r10,r6
	sld	r11,r11,r6
#else
	sld	r10,r10,r6
	sld	r11,r11,r6
	srd	r10,r10,r6
	srd	r11,r11,r6
#endif
	or	r5,r10,r11    /* OR the results to speed things up.  */
	cmpdi	cr7,r5,0      /* If r5 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done)

L(align):
	andi.	r12, r8, 15

	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bne	cr0, L(loop)

	/* Handle WORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	bne	cr7,L(done)
	b	L(loop)	      /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */
	.p2align  5
L(loop):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */
	ld	r12,8(r8)
	ldu	r7,16(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	cmpb	r6,r7,r4
	cmpb	r7,r7,r0
	or	r12,r10,r11
	or	r5,r6,r7
	or	r5,r12,r5
	cmpdi	cr7,r5,0
	beq	cr7,L(vector)

	/* OK, one (or both) of the doublewords contains a c/null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a c/null byte.  */
	cmpdi	cr6,r12,0
	addi	r8,r8,-8
	bne	cr6,L(done)

	/* The c/null byte must be in the second doubleword.  Adjust the
	   address again and move the result of cmpb to r10 so we can calculate
	   the pointer.  */

	mr	r10,r6
	mr	r11,r7
	addi	r8,r8,8

	/* r10/r11 have the output of the cmpb instructions, that is,
	   0xff in the same position as the c/null byte in the original
	   doubleword from the string.  Use that to calculate the pointer.  */

L(done):
	/* If there are more than one 0xff in r11, find the first position of
	   0xff in r11 and fill r10 with 0 from that position.  */
	cmpdi	cr7,r11,0
	beq	cr7,L(no_null)
#ifdef __LITTLE_ENDIAN__
	addi	r3,r11,-1
	andc	r3,r3,r11
	popcntd r0,r3
#else
	cntlzd	r0,r11
#endif
	subfic	r0,r0,63
	li	r6,-1
#ifdef __LITTLE_ENDIAN__
	srd	r0,r6,r0
#else
	sld	r0,r6,r0
#endif
	and	r10,r0,r10
L(no_null):
#ifdef __LITTLE_ENDIAN__
	cntlzd	r0,r10		/* Count leading zeros before c matches.  */
	addi	r3,r10,-1
	andc	r3,r3,r10
	addi	r10,r11,-1
	andc	r10,r10,r11
	cmpld	cr7,r3,r10
	bgt	cr7,L(no_match)
#else
	addi	r3,r10,-1	/* Count trailing zeros before c matches.  */
	andc	r3,r3,r10
	popcntd	r0,r3
	cmpld	cr7,r11,r10
	bgt	cr7,L(no_match)
#endif
	srdi	r0,r0,3		/* Convert trailing zeros to bytes.  */
	subfic	r0,r0,7
	add	r9,r8,r0      /* Return address of the matching c byte
				 or null in case c was not found.  */
	li	r0,0
	cmpdi	cr7,r11,0     /* If r11 == 0, no null's have been found.  */
	beq	cr7,L(align)

	.align	4
L(no_match):
	mr	r3,r9
	blr

/* Check the first 32B in GPR's and move to vectorized loop.  */
	.p2align  5
L(vector):
	addi	r3, r8, 8
	/* Make sure 32B aligned.  */
	andi.	r10, r3, 31
	bne	cr0, L(loop)
	vspltisb	v0, 0
	/* Precompute vbpermq constant.  */
	vspltisb	v10, 3
	lvsl	v11, r0, r0
	vslb	v10, v11, v10
	mtvrd	v1, r4
	li	r5, 16
	vspltb	v1, v1, 7
	/* Compare 32 bytes in each loop.  */
L(continue):
	lvx	v4, 0, r3
	lvx	v5, r3, r5
	vcmpequb	v2, v0, v4
	vcmpequb	v3, v0, v5
	vcmpequb	v6, v1, v4
	vcmpequb	v7, v1, v5
	vor	v8, v2, v3
	vor	v9, v6, v7
	vor	v11, v8, v9
	vcmpequb.	v11, v0, v11
	addi	r3, r3, 32
	blt	cr6, L(continue)
	vcmpequb.	v8, v0, v8
	blt	cr6, L(match)

	/* One (or both) of the quadwords contains c/null.  */
	vspltisb	v8, 2
	vspltisb	v9, 5
	/* Precompute values used for comparison.  */
	vsl	v9, v8, v9	/* v9 = 0x4040404040404040.  */
	vaddubm	v8, v9, v9
	vsldoi	v8, v0, v8, 1	/* v8 = 0x80.  */

	/* Check if null is in second qw.  */
	vcmpequb.	v11, v0, v2
	blt	cr6, L(secondqw)

	/* Null found in first qw.  */
	addi	r8, r3, -32
	/* Calculate the null position.  */
	FIND_NULL_POS(v2)
	/* Check if null is in the first byte.  */
	vcmpequb.	v11, v0, v2
	blt	cr6, L(no_match)
	vsububm	v2, v8, v2
	/* Mask unwanted bytes after null.  */
#ifdef __LITTLE_ENDIAN__
	vslo	v6, v6, v2
	vsro	v6, v6, v2
#else
	vsro	v6, v6, v2
	vslo	v6, v6, v2
#endif
	vcmpequb.	v11, v0, v6
	blt	cr6, L(no_match)
	/* Found a match before null.  */
	CALCULATE_MATCH()
	add	r3, r8, r6
	blr

L(secondqw):
	addi	r8, r3, -16
	FIND_NULL_POS(v3)
	vcmpequb.	v11, v0, v2
	blt	cr6, L(no_match1)
	vsububm	v2, v8, v2
	/* Mask unwanted bytes after null.  */
#ifdef __LITTLE_ENDIAN__
	vslo	v7, v7, v2
	vsro	v7, v7, v2
#else
	vsro	v7, v7, v2
	vslo	v7, v7, v2
#endif
	vcmpequb.	v11, v0, v7
	blt	cr6, L(no_match1)
	addi	r8, r8, 16
	vor	v6, v0, v7
L(no_match1):
	addi	r8, r8, -16
	vcmpequb.	v11, v0, v6
	blt	cr6, L(no_match)
	/* Found a match before null.  */
	CALCULATE_MATCH()
	add	r3, r8, r6
	blr

L(match):
	/* One (or both) of the quadwords contains a match.  */
	mr	r8, r3
	vcmpequb.	v8, v0, v7
	blt	cr6, L(firstqw)
	/* Match found in second qw.  */
	addi	r8, r8, 16
	vor	v6, v0, v7
L(firstqw):
	addi	r8, r8, -32
	CALCULATE_MATCH()
	add	r9, r8, r6      /* Compute final length.  */
	b	L(continue)
/* We are here because strrchr was called with a null byte.  */
	.align	4
L(null_match):
	/* r0 has a doubleword of null bytes.  */

	cmpb	r5,r12,r0     /* Compare each byte against null bytes.  */

	/* Move the doublewords left and right to discard the bits that are
	   not part of the string and bring them back as zeros.  */
#ifdef __LITTLE_ENDIAN__
	srd	r5,r5,r6
	sld	r5,r5,r6
#else
	sld	r5,r5,r6
	srd	r5,r5,r6
#endif
	cmpdi	cr7,r5,0      /* If r5 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done_null)

	andi.	r12, r8, 15

	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bne	cr0, L(loop_null)

	/* Handle WORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb	r5,r12,r0
	cmpdi	cr7,r5,0
	bne	cr7,L(done_null)
	b	L(loop_null)  /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */

	/* Main loop to look for the end of the string.  Since it's a
	   small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop_null):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */
	ld	r12,8(r8)
	ldu	r11,16(r8)
	cmpb	r5,r12,r0
	cmpb	r10,r11,r0
	or	r6,r5,r10
	cmpdi	cr7,r6,0
	beq	cr7,L(vector1)

	/* OK, one (or both) of the doublewords contains a null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a null byte.  */

	cmpdi	cr6,r5,0
	addi	r8,r8,-8
	bne	cr6,L(done_null)

	/* The null byte must be in the second doubleword.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   pointer.  */

	mr	r5,r10
	addi	r8,r8,8

	/* r5 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the null byte in the original
	   doubleword from the string.  Use that to calculate the pointer.  */
L(done_null):
#ifdef __LITTLE_ENDIAN__
	addi	r0,r5,-1
	andc	r0,r0,r5
	popcntd	r0,r0
#else
	cntlzd	r0,r5	      /* Count leading zeros before the match.  */
#endif
	srdi	r0,r0,3	      /* Convert trailing zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching null byte.  */
	blr
/* Check the first 32B in GPR's and move to vectorized loop.  */
	.p2align  5
L(vector1):
	addi	r3, r8, 8
	/* Make sure 32B aligned.  */
	andi.	r10, r3, 31
	bne	cr0, L(loop_null)
	vspltisb	v0, 0
	/* Precompute vbpermq constant.  */
	vspltisb	v10, 3
	lvsl	v11, r0, r0
	vslb	v10, v11, v10
	li	r5, 16
	/* Compare 32 bytes in each loop.  */
L(continue1):
	lvx	v4, 0, r3
	lvx	v5, r3, r5
	vcmpequb	v2, v0, v4
	vcmpequb	v3, v0, v5
	vor	v8, v2, v3
	vcmpequb.	v11, v0, v8
	addi	r3, r3, 32
	blt	cr6, L(continue1)
	addi	r3, r3, -32
	vbpermq	v2, v2, v10
	vbpermq	v3, v3, v10
	/* Shift each component into its correct position for merging.  */
#ifdef __LITTLE_ENDIAN__
	vsldoi	v3, v3, v3, 2
#else
	vsldoi	v2, v2, v2, 6
	vsldoi	v3, v3, v3, 4
#endif
	/* Merge the results and move to a GPR.  */
	vor	v4, v3, v2
	mfvrd	r5, v4
#ifdef __LITTLE_ENDIAN__
	addi	r6, r5, -1
	andc	r6, r6, r5
	popcntd	r6, r6
#else
	cntlzd	r6, r5  /* Count leading zeros before the match.  */
#endif
	add	r3, r3, r6      /* Compute final length.  */
	blr
END_GEN_TB (STRRCHR, TB_TOCLESS)
weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
