/* Optimized strchr implementation for PowerPC64/POWER8.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifdef USE_AS_STRCHRNUL
# ifndef STRCHRNUL
#   define FUNC_NAME __strchrnul
# else
#   define FUNC_NAME STRCHRNUL
# endif
#else
# ifndef STRCHR
#  define FUNC_NAME strchr
# else
#  define FUNC_NAME STRCHR
# endif
#endif  /* !USE_AS_STRCHRNUL  */

/* int [r3] strchr (char *s [r3], int c [r4])  */
	.machine  power8
ENTRY_TOCLESS (FUNC_NAME)
	CALL_MCOUNT 2
	dcbt	0,r3
	clrrdi	r8,r3,3	      /* Align the address to doubleword boundary.  */
	cmpdi	cr7,r4,0
	ld	r12,0(r8)     /* Load doubleword from memory.  */
	li	r0,0	      /* Doubleword with null chars to use
				 with cmpb.  */

	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */

	beq	cr7,L(null_match)

	/* Replicate byte to doubleword.  */
	insrdi	r4,r4,8,48
	insrdi	r4,r4,16,32
	insrdi  r4,r4,32,0

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

	mtcrf   0x01,r8

	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	28,L(loop)

	/* Handle WORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb    r10,r12,r4
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
	ldu	r9,16(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	cmpb	r6,r9,r4
	cmpb	r7,r9,r0
	or	r5,r10,r11
	or	r9,r6,r7
	or	r12,r5,r9
	cmpdi	cr7,r12,0
	beq	cr7,L(vector)
	/* OK, one (or both) of the doublewords contains a c/null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a c/null byte.  */

	cmpdi	cr6,r5,0
	addi	r8,r8,-8
	bne	cr6,L(done)

	/* The c/null byte must be in the second doubleword.  Adjust the
	   address again and move the result of cmpb to r10 so we can calculate
	   the pointer.  */

	mr	r10,r6
	mr	r11,r7
	addi	r8,r8,8
#ifdef USE_AS_STRCHRNUL
	mr	r5, r9
#endif
	/* r10/r11 have the output of the cmpb instructions, that is,
	   0xff in the same position as the c/null byte in the original
	   doubleword from the string.  Use that to calculate the pointer.  */
L(done):
#ifdef USE_AS_STRCHRNUL
	mr	r10, r5
#endif
#ifdef __LITTLE_ENDIAN__
	addi    r3,r10,-1
	andc    r3,r3,r10
	popcntd	r0,r3
# ifndef USE_AS_STRCHRNUL
	addi    r4,r11,-1
	andc    r4,r4,r11
	cmpld	cr7,r3,r4
	bgt	cr7,L(no_match)
# endif
#else
	cntlzd	r0,r10	      /* Count leading zeros before c matches.  */
# ifndef USE_AS_STRCHRNUL
	cmpld	cr7,r11,r10
	bgt	cr7,L(no_match)
# endif
#endif
	srdi	r0,r0,3	      /* Convert leading zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching c byte
				 or null in case c was not found.  */
	blr

	/* Check the first 32B in GPR's and move to vectorized loop.  */
	.p2align  5
L(vector):
	addi	r3, r8, 8
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
	/* One (or both) of the quadwords contains a c/null byte.  */
	addi	r3, r3, -32
#ifndef USE_AS_STRCHRNUL
	vcmpequb.	v11, v0, v9
	blt	cr6, L(no_match)
#endif
	/* Permute the first bit of each byte into bits 48-63.  */
	vbpermq	v2, v2, v10
	vbpermq	v3, v3, v10
	vbpermq	v6, v6, v10
	vbpermq	v7, v7, v10
	/* Shift each component into its correct position for merging.  */
#ifdef __LITTLE_ENDIAN__
	vsldoi	v3, v3, v3, 2
	vsldoi	v7, v7, v7, 2
#else
	vsldoi	v2, v2, v2, 6
	vsldoi	v3, v3, v3, 4
	vsldoi	v6, v6, v6, 6
	vsldoi	v7, v7, v7, 4
#endif

        /* Merge the results and move to a GPR.  */
        vor     v1, v3, v2
        vor     v2, v6, v7
        vor     v4, v1, v2
	mfvrd	r5, v4
#ifdef __LITTLE_ENDIAN__
	addi	r6, r5, -1
	andc	r6, r6, r5
	popcntd	r6, r6
#else
	cntlzd	r6, r5	/* Count leading zeros before the match.  */
#endif
	add	r3, r3, r6	/* Compute final length.  */
	/* Return NULL if null found before c.  */
#ifndef USE_AS_STRCHRNUL
	lbz	r4, 0(r3)
	cmpdi	cr7, r4, 0
	beq	cr7, L(no_match)
#endif
	blr

#ifndef USE_AS_STRCHRNUL
	.align	4
L(no_match):
	li	r3,0
	blr
#endif

/* We are here because strchr was called with a null byte.  */
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
	cmpdi	cr7,r5,0      /* If r10 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done_null)

	mtcrf   0x01,r8

	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	28,L(loop_null)

	/* Handle WORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb    r5,r12,r0
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
	ldu     r11,16(r8)
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
	addi    r0,r5,-1
	andc    r0,r0,r5
	popcntd	r0,r0
#else
	cntlzd	r0,r5	      /* Count leading zeros before the match.  */
#endif
	srdi	r0,r0,3	      /* Convert leading zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching null byte.  */
	blr
	.p2align  5
L(vector1):
	addi    r3, r8, 8
	andi.	r10, r3, 31
	bne	cr0, L(loop_null)
	vspltisb	v8, -1
	vspltisb	v0, 0
	vspltisb	v10, 3
	lvsl	v11, r0, r0
	vslb	v10, v11, v10
	li	r5, 16
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
L(end1):
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
        vor     v4, v3, v2
	mfvrd	r5, v4
#ifdef __LITTLE_ENDIAN__
	addi	r6, r5, -1
	andc	r6, r6, r5
	popcntd	r6, r6
#else
	cntlzd	r6, r5	/* Count leading zeros before the match.  */
#endif
	add	r3, r3, r6	/* Compute final length.  */
	blr
END (FUNC_NAME)

#ifndef USE_AS_STRCHRNUL
weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
#endif
