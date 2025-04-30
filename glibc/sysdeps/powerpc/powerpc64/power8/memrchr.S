/* Optimized memrchr implementation for PowerPC64/POWER8.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   Contributed by Luis Machado <luisgpm@br.ibm.com>.
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

/* int [r3] memrchr (char *s [r3], int byte [r4], int size [r5])  */

#ifndef MEMRCHR
# define MEMRCHR __memrchr
#endif
	.machine  power8
ENTRY_TOCLESS (MEMRCHR)
	CALL_MCOUNT 3
	add	r7, r3, r5      /* Calculate the last acceptable address.  */
	neg	r0, r7
	addi	r7, r7, -1
	mr	r10, r3
	clrrdi	r6, r7, 7
	li	r9, 3<<5
	dcbt	r9, r6, 8       /* Stream hint, decreasing addresses.  */

	/* Replicate BYTE to doubleword.  */
	insrdi	r4, r4, 8, 48
	insrdi	r4, r4, 16, 32
	insrdi	r4, r4, 32, 0
	li	r6, -8
	li	r9, -1
	rlwinm	r0, r0, 3, 26, 28 /* Calculate padding.  */
	clrrdi	r8, r7, 3
	srd	r9, r9, r0
	cmpldi	r5, 32
	clrrdi	r0, r10, 3
	ble	L(small_range)

#ifdef __LITTLE_ENDIAN__
	ldx	r12, 0, r8
#else
	ldbrx	r12, 0, r8      /* Load reversed doubleword from memory.  */
#endif
	cmpb	r3, r12, r4     /* Check for BYTE in DWORD1.  */
	and	r3, r3, r9
	cmpldi	cr7, r3, 0      /* If r3 == 0, no BYTEs have been found.  */
	bne	cr7, L(done)

	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */
	andi.	r12, r8, 15
	beq	cr0, L(align_qw)

	/* Handle DWORD2 of pair.  */
#ifdef __LITTLE_ENDIAN__
	ldx	r12, r8, r6
#else
	ldbrx	r12, r8, r6
#endif
	addi	r8, r8, -8
	cmpb	r3, r12, r4
	cmpldi	cr7, r3, 0
	bne	cr7, L(done)

	.align	4
	/* At this point, r8 is 16B aligned.  */
L(align_qw):
	sub	r5, r8, r0
	vspltisb	v0, 0
	/* Precompute vbpermq constant.  */
	vspltisb	v10, 3
	li	r0, 0
	lvsl	v11, r0, r0
	vslb	v10, v11, v10
	mtvrd	v1, r4
	vspltb	v1, v1, 7
	cmpldi	r5, 64
	ble	L(tail64)
	/* Are we 64-byte aligned? If so, jump to the vectorized loop.
	   Note: aligning to 64-byte will necessarily slow down performance for
	   strings around 64 bytes in length due to the extra comparisons
	   required to check alignment for the vectorized loop.  This is a
	   necessary tradeoff we are willing to take in order to speed up the
	   calculation for larger strings.  */
	andi.	r11, r8, 63
	beq	cr0, L(preloop_64B)
	/* In order to begin the 64B loop, it needs to be 64
	   bytes aligned.  So read until it is 64B aligned.  */
	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	addi	r5, r5, -16

	andi.	r11, r8, 63
	beq	cr0, L(preloop_64B)
	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	addi	r5, r5, -16

	andi.	r11, r8, 63
	beq	cr0, L(preloop_64B)
	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	addi	r5, r5, -16
	/* At this point it should be 64B aligned.
	   Prepare for the 64B loop.  */
L(preloop_64B):
	cmpldi	r5, 64		/* Check if r5 < 64.  */
	ble	L(tail64)
	srdi	r9, r5, 6	/* Number of loop iterations.  */
	mtctr	r9		/* Setup the counter.  */
	li	r11, 16		/* Load required offsets.  */
	li	r9, 32
	li	r7, 48

	/* Handle r5 > 64.  Loop over the bytes in strides of 64B.  */
	.align 4
L(loop):
	addi	r8, r8, -64	/* Adjust address for the next iteration.  */
	lvx	v2, 0, r8	/* Load 4 quadwords.  */
	lvx	v3, r8, r11
	lvx	v4, v8, r9
	lvx	v5, v8, r7
	vcmpequb	v6, v1, v2
	vcmpequb	v7, v1, v3
	vcmpequb	v8, v1, v4
	vcmpequb	v9, v1, v5
	vor	v11, v6, v7
	vor	v12, v8, v9
	vor	v11, v11, v12	/* Compare and merge into one VR for speed.  */
	vcmpequb.	v11, v0, v11
	bnl	cr6, L(found)
	bdnz	L(loop)
	clrldi	r5, r5, 58

	/* Handle remainder of 64B loop or r5 > 64.  */
	.align	4
L(tail64):
	cmpldi	r5, 0
	beq	L(null)
	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	cmpldi	cr6, r5, 16
	ble	cr6, L(null)
	addi	r5, r5, -16

	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	cmpldi	cr6, r5, 16
	ble	cr6, L(null)
	addi	r5, r5, -16

	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	cmpldi	cr6, r5, 16
	ble	cr6, L(null)
	addi	r5, r5, -16

	addi	r8, r8, -16
	lvx	v4, 0, r8
	vcmpequb	v6, v1, v4
	vcmpequb.	v11, v0, v6
	bnl	cr6, L(found_16B)
	li	r3, 0
	blr

	/* Found a match in 64B loop.  */
	.align	4
L(found):
	/* Permute the first bit of each byte into bits 48-63.  */
	vbpermq	v6, v6, v10
	vbpermq	v7, v7, v10
	vbpermq	v8, v8, v10
	vbpermq	v9, v9, v10
	/* Shift each component into its correct position for merging.  */
#ifdef __LITTLE_ENDIAN__
	vsldoi	v7, v7, v7, 2
	vsldoi	v8, v8, v8, 4
	vsldoi	v9, v9, v9, 6
#else
	vsldoi	v6, v6, v6, 6
	vsldoi	v7, v7, v7, 4
	vsldoi	v8, v8, v8, 2
#endif
	/* Merge the results and move to a GPR.  */
	vor	v11, v6, v7
	vor	v4, v9, v8
	vor	v4, v11, v4
	mfvrd	r5, v4
#ifdef __LITTLE_ENDIAN__
	cntlzd	r6, r5	/* Count leading zeros before the match.  */
#else
	addi	r6, r5, -1
	andc	r6, r6, r5
	popcntd	r6, r6
#endif
	addi	r8, r8, 63
	sub	r3, r8, r6	/* Compute final address.  */
	cmpld	cr7, r3, r10
	bgelr	cr7
	li	r3, 0
	blr

	/* Found a match in last 16 bytes.  */
	.align	4
L(found_16B):
	cmpld	r8, r10		/* Are we on the last QW?  */
	bge	L(last)
	/* Now discard bytes before starting address.  */
	sub	r9, r10, r8
	mtvrd	v9, r9
	vspltisb	v8, 3
	/* Mask unwanted bytes.  */
#ifdef __LITTLE_ENDIAN__
	lvsr	v7, 0, r10
	vperm   v6, v0, v6, v7
	vsldoi	v9, v0, v9, 8
	vsl	v9, v9, v8
	vslo	v6, v6, v9
#else
	lvsl	v7, 0, r10
	vperm   v6, v6, v0, v7
	vsldoi	v9, v0, v9, 8
	vsl	v9, v9, v8
	vsro	v6, v6, v9
#endif
L(last):
	/* Permute the first bit of each byte into bits 48-63.  */
	vbpermq	v6, v6, v10
	/* Shift each component into its correct position for merging.  */
#ifdef __LITTLE_ENDIAN__
	vsldoi	v6, v6, v6, 6
	mfvrd	r7, v6
	cntlzd	r6, r7	/* Count leading zeros before the match.  */
#else
	mfvrd	r7, v6
	addi	r6, r7, -1
	andc	r6, r6, r7
	popcntd	r6, r6
#endif
	addi	r8, r8, 15
	sub	r3, r8, r6	/* Compute final address.  */
	cmpld	r6, r5
	bltlr
	li	r3, 0
	blr

	/* r3 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as BYTE in the original
	   word from the string.  Use that to calculate the pointer.
	   We need to make sure BYTE is *before* the end of the
	   range.  */
L(done):
	cntlzd	r9, r3	      /* Count leading zeros before the match.  */
	cmpld	r8, r0         /* Are we on the last word?  */
	srdi	r6, r9, 3	      /* Convert leading zeros to bytes.  */
	addi	r0, r6, -7
	sub	r3, r8, r0
	cmpld	cr7, r3, r10
	bnelr
	bgelr	cr7
	li	r3, 0
	blr

	.align	4
L(null):
	li	r3, 0
	blr

/* Deals with size <= 32.  */
	.align	4
L(small_range):
	cmpldi	r5, 0
	beq	L(null)

#ifdef __LITTLE_ENDIAN__
	ldx	r12, 0, r8
#else
	ldbrx	r12, 0, r8      /* Load reversed doubleword from memory.  */
#endif
	cmpb	r3, r12, r4     /* Check for BYTE in DWORD1.  */
	and	r3, r3, r9
	cmpldi	cr7, r3, 0
	bne	cr7, L(done)

	/* Are we done already?  */
	cmpld	r8, r0
	addi	r8, r8, -8
	beqlr

	.align	5
L(loop_small):
#ifdef __LITTLE_ENDIAN__
	ldx	r12, 0, r8
#else
	ldbrx	r12, 0, r8
#endif
	cmpb	r3, r12, r4
	cmpld	r8, r0
	cmpldi	cr7, r3, 0
	bne	cr7, L(done)
	addi	r8, r8, -8
	bne	L(loop_small)
	blr

END (MEMRCHR)
weak_alias (__memrchr, memrchr)
libc_hidden_builtin_def (memrchr)
