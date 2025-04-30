/* Optimized strnlen implementation for POWER8 using a vmx loop.

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

/* It is implemented the following heuristic:
	1. Case maxlen <= 32: align the pointer to 8 bytes to loop through
	reading doublewords. Uses the POWER7 algorithm.
	2. Case maxlen > 32: check for null bytes in the first 16 bytes using
	unaligned accesses. Return length if found. Otherwise:
		2.1 Case maxlen < 64: deduct the bytes previously read, align
		the pointer to 16 bytes and loop through reading quadwords
		until find null bytes or reach maxlen.
		2.2 Case maxlen > 64: deduct the bytes previously read, align
		the pointer to 64 bytes and set up a counter to loop through
		reading in strides of 64 bytes. In case it finished the loop
		with null bytes not found, process the remainder bytes by
		switching to the loop to heuristic in 2.1.  */

#include <sysdep.h>

/* Define default page size to 4KB.  */
#define PAGE_SIZE 4096


/* int [r3] strnlen (char *s [r3], size_t maxlen [r4])  */
	.machine  power8
ENTRY_TOCLESS (__strnlen)
	CALL_MCOUNT 2
	dcbt	0,r3

	cmpldi	r4,32           /* Check if maxlen <= 32.  */
	ble	L(small_range)  /* If maxlen <= 32.  */

	/* Upcoming 16 bytes unaligned accesses cannot cross the page boundary
	   otherwise the processor throws an memory access error.
	   Use following code to check there is room for such as accesses:
	     (((size_t) s) % PAGE_SIZE > (PAGE_SIZE - 16)
	   If it is disallowed then switch to the code that handles
	   the string when maxlen <= 32.  */
	clrldi	r10,r3,52
	cmpldi  cr7,r10,PAGE_SIZE-16
	bgt     cr7,L(small_range)	/* If less than 16B of page end.  */

	/* Compute our permute constant r8.  */
	li	r7,0
	/* Compute a bpermd constant to move bit 0 of each word into
	   a halfword value, and count trailing zeros.  */
#ifdef __LITTLE_ENDIAN__
	li	r8,0x2820
	oris	r8,r8,0x3830
	sldi	r8,r8,32
	ori	r8,r8,0x0800
	oris	r8,r8,0x1810
#else
	li	r8,0x1018
	oris	r8,r8,0x0008
	sldi	r8,r8,32
	ori	r8,r8,0x3038
	oris	r8,r8,0x2028
#endif

	/* maxlen > 32. Optimistically check for null bytes in the first
	   16 bytes of the string using unaligned accesses.  */
	ld	r5,0(r3)
	ld	r6,8(r3)
	cmpb	r10,r7,r5		/* Check for null bytes in DWORD1.  */
	cmpb	r11,r7,r6		/* Check for null bytes in DWORD2.  */
	or.	r7,r10,r11
	bne	cr0, L(early_find)	/* If found null bytes.  */

	/* At this point maxlen > 32 and null bytes were not found at first
	   16 bytes. Prepare for loop using VMX.  */

	/* r3 == s, r4 == maxlen. All other volatile regs are unused now.  */

	addi	r5,r3,16	/* Align up, or just add the 16B we
				   already checked.  */
	li	r0,15
	and	r7,r5,r0	/* Find offset into 16B alignment.  */
	andc	r5,r5,r0	/* Quadword align up s to the next quadword.  */
	li	r0,16
	subf	r0,r7,r0
	subf	r4,r0,r4	/* Deduct unaligned bytes from maxlen.  */


	/* Compute offsets for vmx loads, and precompute the vbpermq
	   constants for both the 64B and 16B loops.  */
	li	r6,0
	vspltisb  v0,0
	vspltisb  v10,3
	lvsl	  v11,r6,r6
	vslb	  v10,v11,v10

	cmpldi  r4,64		/* Check maxlen < 64.  */
	blt	L(smaller)	/* If maxlen < 64 */

	/* In order to begin the 64B loop, it needs to be 64
	   bytes aligned. So read quadwords until it is aligned or found null
	   bytes. At worst case it will be aligned after the fourth iteration,
	   so unroll the loop to avoid counter checking.  */
	andi.   r7,r5,63		/* Check if is 64 bytes aligned.  */
	beq     cr0,L(preloop_64B)	/* If it is already 64B aligned.  */
	lvx     v1,r5,r6
	vcmpequb.       v1,v1,v0
	addi    r5,r5,16
	addi    r4,r4,-16		/* Decrement maxlen in 16 bytes. */
	bne     cr6,L(found_aligning64B) /* If found null bytes.  */

	/* Unroll 2x above code block until aligned or find null bytes.  */
	andi.   r7,r5,63
	beq     cr0,L(preloop_64B)
	lvx     v1,r5,r6
	vcmpequb.      v1,v1,v0
	addi    r5,r5,16
	addi    r4,r4,-16
	bne     cr6,L(found_aligning64B)

	andi.   r7,r5,63
	beq     cr0,L(preloop_64B)
	lvx     v1,r5,r6
	vcmpequb.      v1,v1,v0
	addi    r5,r5,16
	addi    r4,r4,-16
	bne     cr6,L(found_aligning64B)

	/* At this point it should be 16 bytes aligned.
	   Prepare for the 64B loop.  */
	.p2align 4
L(preloop_64B):
	/* Check if maxlen became is less than 64, therefore disallowing the
	   64B loop. If it happened switch to the 16B loop code.  */
	cmpldi  r4,64		/* Check if maxlen < 64.  */
	blt     L(smaller)	/* If maxlen < 64.  */
	/* Set some constant values.  */
	li      r7,16
	li      r10,32
	li      r9,48

	/* Compute the number of 64 bytes iterations needed.  */
	srdi	r11,r4,6	/* Compute loop count (maxlen / 64).  */
	andi.	r4,r4,63	/* Set maxlen the remainder (maxlen % 64).  */
	mtctr	r11		/* Move loop count to counter register.  */

	/* Handle maxlen > 64. Loop over the bytes in strides of 64B.  */
	.p2align 4
L(loop_64B):
	lvx	v1,r5,r6	/* r5 is the pointer to s.  */
	lvx	v2,r5,r7
	lvx	v3,r5,r10
	lvx	v4,r5,r9
	/* Compare the four 16B vectors to obtain the least 16 values.
	   Null bytes should emerge into v7, then check for null bytes.  */
	vminub	v5,v1,v2
	vminub	v6,v3,v4
	vminub	v7,v5,v6
	vcmpequb. v7,v7,v0		/* Check for null bytes.  */
	addi	r5,r5,64		/* Add pointer to next iteraction.  */
	bne	cr6,L(found_64B)	/* If found null bytes.  */
	bdnz	L(loop_64B)		/* Continue the loop if count > 0. */

/* Hit loop end without null match. So branch to handle the remainder.  */

	/* Prepare a 16B loop to handle two cases:
		1. If 32 > maxlen < 64.
		2. If maxlen >= 64, and reached end of the 64B loop with null
		bytes not found. Thus handle the remainder bytes here. */
	.p2align 4
L(smaller):
        cmpldi  r4,0            /* Check maxlen is zero.  */
        beq     L(done)         /* If maxlen is zero.  */

	/* Place rounded up number of qw's to check into a vmx
	   register, and use some vector tricks to minimize
	   branching.  */
        mtvrd    v7,r4          /* copy maxlen from gpr to vector register. */
        vspltisb v5,1
        vspltisb v6,15
        vspltb   v2,v7,7
        vaddubs  v3,v5,v6

#ifdef __LITTLE_ENDIAN__
	vspltish v5,1           /* Compute 16 in each byte.  */
#endif

	/* Loop in 16B aligned incremements now. */
	.p2align 4
L(loop_16B):
	lvx     v1,r5,r6        /* Load quadword into vector register.  */
	addi    r5,r5,16        /* Increment address to next 16B block.  */
	vor     v7,v2,v2        /* Save loop count (v2) into v7. */
	vsububs v2,v2,v3        /* Subtract 16B from count, saturate at 0. */
	vminub  v4,v1,v2
	vcmpequb. v4,v4,v0      /* Checking for null bytes.  */
	beq     cr6,L(loop_16B) /* If null bytes not found.  */

	vcmpequb  v1,v1,v0
	vbpermq   v1,v1,v10
#ifdef __LITTLE_ENDIAN__
	vsubuhm  v2,v1,v5       /* Form a mask of trailing zeros.  */
	vandc    v2,v2,v1
	vpopcnth v1,v2          /* count of trailing zeros, 16 if none.  */
#else
	vclzh    v1,v1          /* count the leading zeros, 16 if none.  */
#endif
	/* Truncate to maximum allowable offset.  */
	vcmpgtub v2,v1,v7       /* Compare and truncate for matches beyond
				   maxlen.  */
	vsel     v1,v1,v7,v2    /* 0-16 is now in byte 7.  */

	mfvrd   r0,v1
	addi    r5,r5,-16       /* Undo speculative bump.  */
	extsb   r0,r0           /* Clear whatever gunk is in the high 56b.  */
	add     r5,r5,r0        /* Add the offset of whatever was found.  */
L(done):
	subf    r3,r3,r5        /* Length is equal to the offset of null byte
				   matched minus the pointer to s.  */
	blr                     /* Done.  */

	/* Handle case of maxlen > 64 and found null bytes in last block
	   of 64 bytes read.  */
	.p2align 4
L(found_64B):
	/* A zero was found. Reduce the result.  */
	vcmpequb  v1,v1,v0
	vcmpequb  v2,v2,v0
	vcmpequb  v3,v3,v0
	vcmpequb  v4,v4,v0

	/* Permute the first bit of each byte into bits 48-63.  */
	vbpermq	v1,v1,v10
	vbpermq	v2,v2,v10
	vbpermq	v3,v3,v10
	vbpermq	v4,v4,v10

	/* Shift each component into its correct position for merging.  */
#ifdef __LITTLE_ENDIAN__
	vsldoi	v2,v2,v2,2
	vsldoi	v3,v3,v3,4
	vsldoi	v4,v4,v4,6
#else
	vsldoi	v1,v1,v1,6
	vsldoi	v2,v2,v2,4
	vsldoi	v3,v3,v3,2
#endif

	/* Merge the results and move to a GPR.  */
	vor	v1,v2,v1
	vor	v2,v3,v4
	vor	v4,v1,v2

	/* Adjust address to the start of the current 64B block.  */
	addi	r5,r5,-64

	mfvrd	r10,v4
#ifdef __LITTLE_ENDIAN__
	addi	r9,r10,-1	/* Form a mask from trailing zeros.  */
	andc	r9,r9,r10
	popcntd	r0,r9		/* Count the bits in the mask.  */
#else
	cntlzd	r0,r10		/* Count leading zeros before the match.  */
#endif
	subf	r5,r3,r5
	add	r3,r5,r0	/* Compute final length.  */
	blr                     /* Done.  */

	/* Handle case where null bytes were found while aligning
	   as a preparation for the 64B loop.  */
	.p2align 4
L(found_aligning64B):
	vbpermq v1,v1,v10
#ifdef __LITTLE_ENDIAN__
	mfvrd   r10,v1
	addi    r9,r10,-1       /* Form a mask from trailing zeros.  */
	andc    r9,r9,r10
	popcntd r0,r9           /* Count the bits in the mask.  */
#else
	vsldoi  v1,v1,v1,6
	mfvrd   r10,v1
	cntlzd  r0,r10          /* Count leading zeros before the match.  */
#endif
	addi    r5,r5,-16	/* Adjust address to offset of last 16 bytes
				   read.  */
	/* Calculate length as subtracted the pointer to s of last 16 bytes
	   offset, added with the bytes before the match.  */
	subf    r5,r3,r5
	add     r3,r5,r0
	blr			/* Done.  */

	/* Handle case of maxlen > 32 and found a null bytes within the first
	   16 bytes of s.  */
	.p2align 4
L(early_find):
	bpermd	r5,r8,r10        /* r8 contains the bit permute constants.  */
	bpermd	r6,r8,r11
	sldi	r5,r5,8
	or	r5,r5,r6	/* r5 should hold a 16B mask of
				   a potential 0.  */
	cntlzd	r5,r5		/* Count leading zeros.  */
	addi	r3,r5,-48	/* Deduct the 48 leading zeros always
				   present.  */
	blr			/* Done.  */

	/* Handle case of maxlen <= 32. Use the POWER7 algorithm.  */
	.p2align 4
L(small_range):
	clrrdi	r8,r3,3  	/* Align the pointer to 8B.  */
	li	r0,0
	/* Register's content at this point:
	   r3 == pointer to s, r4 == maxlen, r8 == pointer to s aligned to 8B,
	   r7 == last acceptable address. */
	cmpldi	r4,0                 /* Check if maxlen is zero.  */
	beq	L(end_max)	     /* If maxlen is zero.  */

	/* Calculate the last acceptable address and check for possible
	   addition overflow by using satured math:
	   r7 = r3 + r4
	   r7 |= -(r7 < x)  */
	add     r7,r3,r4
	subfc   r6,r3,r7
	subfe   r9,r9,r9
	extsw   r6,r9
	or      r7,r7,r6
	addi    r7,r7,-1

	clrrdi	r7,r7,3              /* Align to 8B address of last
					acceptable address.  */

	rlwinm	r6,r3,3,26,28        /* Calculate padding.  */
	ld	r12,0(r8)            /* Load aligned doubleword.  */
	cmpb	r10,r12,r0           /* Check for null bytes. */
#ifdef __LITTLE_ENDIAN__
	srd	r10,r10,r6
	sld	r10,r10,r6
#else
	sld	r10,r10,r6
	srd	r10,r10,r6
#endif /* __LITTLE_ENDIAN__  */
	cmpldi	cr7,r10,0
	bne	cr7,L(done_small)    /* If found null byte.  */

	cmpld	r8,r7                /* Check if reached maxlen.  */
	beq	L(end_max)	     /* If reached maxlen.  */

	/* Still handling case of maxlen <= 32. Read doubleword aligned until
	   find null bytes or reach maxlen.  */
	.p2align 4
L(loop_small):
	ldu	r12,8(r8)         /* Load next doubleword and update r8.  */
	cmpb	r10,r12,r0        /* Check for null bytes.  */
	cmpldi	cr6,r10,0
	bne	cr6,L(done_small) /* If found null bytes.  */
	cmpld	r8,r7             /* Check if reached maxlen. */
	bne	L(loop_small)	  /* If it has more bytes to read.  */
	mr	r3,r4             /* Reached maxlen with null bytes not found.
				     Length is equal to maxlen.  */
	blr			  /* Done.  */

	/* Still handling case of maxlen <= 32. Found null bytes.
	   Registers: r10 == match bits within doubleword, r8 == address of
	   last doubleword read, r3 == pointer to s, r4 == maxlen.  */
	.p2align 4
L(done_small):
#ifdef __LITTLE_ENDIAN__
	/* Count trailing zeros.  */
	addi	r0,r10,-1
	andc	r0,r0,r10
	popcntd	r0,r0
#else
	cntlzd	r0,r10	      /* Count leading zeros before the match.  */
#endif
	sub	r3,r8,r3      /* Calculate total of bytes before the match.  */
	srdi	r0,r0,3	      /* Convert leading/trailing zeros to bytes.  */
	add	r3,r3,r0      /* Length until the match.  */
	cmpld	r3,r4         /* Check length is greater than maxlen.  */
	blelr
	mr	r3,r4	      /* If length is greater than maxlen, return
				 maxlen.  */
	blr

	/* Handle case of reached maxlen with null bytes not found.  */
	.p2align 4
L(end_max):
	mr	r3,r4	/* Length is equal to maxlen.  */
	blr		/* Done.  */


END (__strnlen)
libc_hidden_def (__strnlen)
weak_alias (__strnlen, strnlen)
libc_hidden_def (strnlen)
