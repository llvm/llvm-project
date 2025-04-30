/* Optimized strlen implementation for PowerPC64/POWER8 using a vectorized
   loop.
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

/* int [r3] strlen (char *s [r3])  */

#ifndef STRLEN
# define STRLEN strlen
#endif
	.machine  power8
ENTRY_TOCLESS (STRLEN, 4)
	CALL_MCOUNT 1
	dcbt	0,r3
	clrrdi	r4,r3,3	      /* Align the address to doubleword boundary.  */
	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */
	li	r0,0	      /* Doubleword with null chars to use
				 with cmpb.  */
	li	r5,-1	      /* MASK = 0xffffffffffffffff.  */
	ld	r12,0(r4)     /* Load doubleword from memory.  */
#ifdef __LITTLE_ENDIAN__
	sld	r5,r5,r6
#else
	srd	r5,r5,r6      /* MASK = MASK >> padding.  */
#endif
	orc	r9,r12,r5     /* Mask bits that are not part of the string.  */
	cmpb	r10,r9,r0     /* Check for null bytes in DWORD1.  */
	cmpdi	cr7,r10,0     /* If r10 == 0, no null's have been found.  */
	bne	cr7,L(done)

	/* For shorter strings (< 64 bytes), we will not use vector registers,
	   as the overhead isn't worth it.  So, let's use GPRs instead.  This
	   will be done the same way as we do in the POWER7 implementation.
	   Let's see if we are aligned to a quadword boundary.  If so, we can
	   jump to the first (non-vectorized) loop.  Otherwise, we have to
	   handle the next DWORD first.  */
	mtcrf	0x01,r4
	mr	r9,r4
	addi	r9,r9,8
	bt	28,L(align64)

	/* Handle the next 8 bytes so we are aligned to a quadword
	   boundary.  */
	ldu	r5,8(r4)
	cmpb	r10,r5,r0
	cmpdi	cr7,r10,0
	addi	r9,r9,8
	bne	cr7,L(done)

L(align64):
	/* Proceed to the old (POWER7) implementation, checking two doublewords
	   per iteraction.  For the first 56 bytes, we will just check for null
	   characters.  After that, we will also check if we are 64-byte aligned
	   so we can jump to the vectorized implementation.  We will unroll
	   these loops to avoid excessive branching.  */
	ld	r6,8(r4)
	ldu	r5,16(r4)
	cmpb	r10,r6,r0
	cmpb	r11,r5,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	addi	r9,r9,16
	bne	cr7,L(dword_zero)

	ld	r6,8(r4)
	ldu	r5,16(r4)
	cmpb	r10,r6,r0
	cmpb	r11,r5,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	addi	r9,r9,16
	bne	cr7,L(dword_zero)

	ld	r6,8(r4)
	ldu	r5,16(r4)
	cmpb	r10,r6,r0
	cmpb	r11,r5,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	addi	r9,r9,16
	bne	cr7,L(dword_zero)

	/* Are we 64-byte aligned? If so, jump to the vectorized loop.
	   Note: aligning to 64-byte will necessarily slow down performance for
	   strings around 64 bytes in length due to the extra comparisons
	   required to check alignment for the vectorized loop.  This is a
	   necessary tradeoff we are willing to take in order to speed up the
	   calculation for larger strings.  */
	andi.	r10,r9,63
	beq	cr0,L(preloop)
	ld	r6,8(r4)
	ldu	r5,16(r4)
	cmpb	r10,r6,r0
	cmpb	r11,r5,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	addi	r9,r9,16
	bne	cr7,L(dword_zero)

	andi.	r10,r9,63
	beq	cr0,L(preloop)
	ld	r6,8(r4)
	ldu	r5,16(r4)
	cmpb	r10,r6,r0
	cmpb	r11,r5,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	addi	r9,r9,16
	bne	cr7,L(dword_zero)

	andi.	r10,r9,63
	beq	cr0,L(preloop)
	ld	r6,8(r4)
	ldu	r5,16(r4)
	cmpb	r10,r6,r0
	cmpb	r11,r5,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	addi	r9,r9,16

	/* At this point, we are necessarily 64-byte aligned.  If no zeroes were
	   found, jump to the vectorized loop.  */
	beq	cr7,L(preloop)

L(dword_zero):
	/* OK, one (or both) of the doublewords contains a null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a null byte.  */

	cmpdi	cr6,r10,0
	addi	r4,r4,-8
	bne	cr6,L(done)

	/* The null byte must be in the second doubleword.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   length.  */

	mr	r10,r11
	addi	r4,r4,8

	/* If the null byte was found in the non-vectorized code, compute the
	   final length.  r10 has the output of the cmpb instruction, that is,
	   it contains 0xff in the same position as the null byte in the
	   original doubleword from the string.  Use that to calculate the
	   length.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi	r9, r10,-1    /* Form a mask from trailing zeros.  */
	andc	r9, r9,r10
	popcntd	r0, r9	      /* Count the bits in the mask.  */
#else
	cntlzd	r0,r10	      /* Count leading zeros before the match.  */
#endif
	subf	r5,r3,r4
	srdi	r0,r0,3	      /* Convert leading/trailing zeros to bytes.  */
	add	r3,r5,r0      /* Compute final length.  */
	blr

	/* Vectorized implementation starts here.  */
	.p2align  4
L(preloop):
	/* Set up for the loop.  */
	mr	r4,r9
	li	r7, 16	      /* Load required offsets.  */
	li	r8, 32
	li	r9, 48
	li	r12, 8
	vxor	v0,v0,v0      /* VR with null chars to use with
				 vcmpequb.  */

	/* Main loop to look for the end of the string.  We will read in
	   64-byte chunks.  Align it to 32 bytes and unroll it 3 times to
	   leverage the icache performance.  */
	.p2align  5
L(loop):
	lvx	  v1,r4,r0  /* Load 4 quadwords.  */
	lvx	  v2,r4,r7
	lvx	  v3,r4,r8
	lvx	  v4,r4,r9
	vminub	  v5,v1,v2  /* Compare and merge into one VR for speed.  */
	vminub	  v6,v3,v4
	vminub	  v7,v5,v6
	vcmpequb. v7,v7,v0  /* Check for NULLs.  */
	addi	  r4,r4,64  /* Adjust address for the next iteration.  */
	bne	  cr6,L(vmx_zero)

	lvx	  v1,r4,r0  /* Load 4 quadwords.  */
	lvx	  v2,r4,r7
	lvx	  v3,r4,r8
	lvx	  v4,r4,r9
	vminub	  v5,v1,v2  /* Compare and merge into one VR for speed.  */
	vminub	  v6,v3,v4
	vminub	  v7,v5,v6
	vcmpequb. v7,v7,v0  /* Check for NULLs.  */
	addi	  r4,r4,64  /* Adjust address for the next iteration.  */
	bne	  cr6,L(vmx_zero)

	lvx	  v1,r4,r0  /* Load 4 quadwords.  */
	lvx	  v2,r4,r7
	lvx	  v3,r4,r8
	lvx	  v4,r4,r9
	vminub	  v5,v1,v2  /* Compare and merge into one VR for speed.  */
	vminub	  v6,v3,v4
	vminub	  v7,v5,v6
	vcmpequb. v7,v7,v0  /* Check for NULLs.  */
	addi	  r4,r4,64  /* Adjust address for the next iteration.  */
	beq	  cr6,L(loop)

L(vmx_zero):
	/* OK, we found a null byte.  Let's look for it in the current 64-byte
	   block and mark it in its corresponding VR.  */
	vcmpequb  v1,v1,v0
	vcmpequb  v2,v2,v0
	vcmpequb  v3,v3,v0
	vcmpequb  v4,v4,v0

	/* We will now 'compress' the result into a single doubleword, so it
	   can be moved to a GPR for the final calculation.  First, we
	   generate an appropriate mask for vbpermq, so we can permute bits into
	   the first halfword.  */
	vspltisb  v10,3
	lvsl	  v11,r0,r0
	vslb	  v10,v11,v10

	/* Permute the first bit of each byte into bits 48-63.  */
	vbpermq	v1,v1,v10
	vbpermq	v2,v2,v10
	vbpermq	v3,v3,v10
	vbpermq	v4,v4,v10

	/* Shift each component into its correct position for merging.  */
#ifdef __LITTLE_ENDIAN__
	vsldoi  v2,v2,v2,2
	vsldoi  v3,v3,v3,4
	vsldoi  v4,v4,v4,6
#else
	vsldoi	v1,v1,v1,6
	vsldoi	v2,v2,v2,4
	vsldoi	v3,v3,v3,2
#endif

	/* Merge the results and move to a GPR.  */
	vor	v1,v2,v1
	vor	v2,v3,v4
	vor	v4,v1,v2
	mfvrd	r10,v4

	 /* Adjust address to the begninning of the current 64-byte block.  */
	addi	r4,r4,-64

#ifdef __LITTLE_ENDIAN__
	addi	r9, r10,-1    /* Form a mask from trailing zeros.  */
	andc	r9, r9,r10
	popcntd	r0, r9	      /* Count the bits in the mask.  */
#else
	cntlzd	r0,r10	      /* Count leading zeros before the match.  */
#endif
	subf	r5,r3,r4
	add	r3,r5,r0      /* Compute final length.  */
	blr

END (STRLEN)
libc_hidden_builtin_def (strlen)
