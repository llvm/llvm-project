/* Optimized strspn implementation for Power8.

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

/* size_t [r3] strspn (const char *string [r3],
                       const char *needleAccept [r4])  */

/* This takes a novel approach by computing a 256 bit mask whereby
   each set bit implies the byte is "accepted".  P8 vector hardware
   has extremely efficient hardware for selecting bits from a mask.

   One might ask "why not use bpermd for short strings"?  It is
   so slow that its performance about matches the generic PPC64
   variant without any fancy masking, with the added expense of
   making the mask.  That was the first variant of this.  */



#include "sysdep.h"

#ifndef USE_AS_STRCSPN
#  define USE_AS_STRCSPN 0
#  ifndef STRSPN
#    define STRSPN strspn
#  endif
#  define INITIAL_MASK 0
#  define UPDATE_MASK(RA, RS, RB) or	RA, RS, RB
#else
#  ifndef STRSPN
#    define STRSPN strcspn
#  endif
#  define INITIAL_MASK -1
#  define UPDATE_MASK(RA, RS, RB) andc	RA, RS, RB
#endif

/* Simple macro to use VSX instructions in overlapping VR's.  */
#define XXVR(insn, vrt, vra, vrb) \
	insn 32+vrt, 32+vra, 32+vrb

	.machine  power8
ENTRY_TOCLESS (STRSPN, 4)
	CALL_MCOUNT 2

	/* Generate useful constants for later on.  */
	vspltisb v1, 7
	vspltisb v2, -1
	vslb	v1, v1, v1	/* 0x80 to swap high bit for vbpermq.  */
	vspltisb v10, 0
	vsldoi	v4, v10, v2, 2	/* 0xFFFF into vr4.  */
	XXVR(xxmrgld, v4, v4, v10) /* Mask for checking matches.  */

	/* Prepare to compute 256b mask.  */
	addi	r4, r4, -1
	li	r5, INITIAL_MASK
	li	r6, INITIAL_MASK
	li	r7, INITIAL_MASK
	li	r8, INITIAL_MASK

#if USE_AS_STRCSPN
	/* Ensure the null character never matches by clearing ISA bit 0 in
	   in r5 which is the bit which will check for it in the later usage
	   of vbpermq.  */
	srdi	r5, r5, 1
#endif

	li	r11, 1
	sldi	r11, r11, 63

	/* Start interleaved Mask computation.
	   This will eventually or 1's into ignored bits from vbpermq.  */
	lvsr	v11, 0, r3
	vspltb  v11, v11, 0	/* Splat shift constant.  */

	/* Build a 256b mask in r5-r8.  */
	.align 4
L(next_needle):
	lbzu	r9, 1(r4)

	cmpldi	cr0, r9, 0
	cmpldi	cr1, r9, 128

	/* This is a little tricky.  srd only uses the first 7 bits,
	   and if bit 7 is set, value is always 0.  So, we can
	   effectively shift 128b in this case.  */
	xori	r12, r9,  0x40	/* Invert bit 6.  */
	srd	r10, r11, r9	/* Mask for bits 0-63.  */
	srd	r12, r11, r12	/* Mask for bits 64-127.  */

	beq	cr0, L(start_cmp)

	/* Now, or the value into the correct GPR.  */
	bge cr1,L(needle_gt128)
	UPDATE_MASK (r5, r5, r10)	/* 0 - 63.  */
	UPDATE_MASK (r6, r6, r12)	/* 64 - 127.  */
	b L(next_needle)

	.align 4
L(needle_gt128):
	UPDATE_MASK (r7, r7, r10)	/* 128 - 191.  */
	UPDATE_MASK (r8, r8, r12)	/* 192 - 255.  */
	b L(next_needle)


	.align 4
L(start_cmp):
	/* Move and merge bitmap into 2 VRs.  bpermd is slower on P8.  */
	mr	r0, r3		/* Save r3 for final length computation.  */
	mtvrd	v5, r5
	mtvrd	v6, r6
	mtvrd	v7, r7
	mtvrd	v8, r8

	/* Continue interleaved mask generation.  */
#ifdef __LITTLE_ENDIAN__
	vsrw	v11, v2, v11	/* Note, shift ignores higher order bits.  */
	vsplth  v11, v11, 0	/* Only care about the high 16 bits of v10.  */
#else
	vslw	v11, v2, v11	/* Note, shift ignores higher order bits.  */
	vsplth  v11, v11, 1	/* Only care about the low 16 bits of v10.  */
#endif
	lvx	v0, 0, r3	/* Note, unaligned load ignores lower bits.  */

	/* Do the merging of the bitmask.  */
	XXVR(xxmrghd, v5, v5, v6)
	XXVR(xxmrghd, v6, v7, v8)

	/* Finish mask generation.  */
	vand	v11, v11, v4	/* Throwaway bits not in the mask.  */

	/* Compare the first 1-16B, while masking unwanted bytes.  */
	clrrdi  r3, r3, 4	/* Note,  counts from qw boundaries.  */
	vxor	v9, v0, v1	/* Swap high bit.  */
	vbpermq	v8, v5, v0
	vbpermq	v7, v6, v9
	vor	v7, v7, v8
	vor	v7, v7, v11	/* Ignore non-participating bytes.  */
	vcmpequh. v8, v7, v4
	bnl	cr6, L(done)

	addi	r3, r3, 16

	.align 4
L(vec):
	lvx	v0, 0, r3
	addi	r3, r3, 16
	vxor	v9, v0, v1	/* Swap high bit.  */
	vbpermq	v8, v5, v0
	vbpermq	v7, v6, v9
	vor	v7, v7, v8
	vcmpequh. v8, v7, v4
	blt	cr6, L(vec)

	addi	r3, r3, -16
L(done):
	subf	r3, r0, r3
	mfvrd	r10, v7

#ifdef __LITTLE_ENDIAN__
	addi	r0,  r10, 1	/* Count the trailing 1's.  */
	andc	r10, r10, r0
	popcntd	r10, r10
#else
	xori	r10, r10, 0xffff /* Count leading 1's by inverting.  */
	addi	r3,  r3,  -48	/* Account for the extra leading zeros.  */
	cntlzd  r10, r10
#endif

	add	r3, r3, r10
	blr

END(STRSPN)
libc_hidden_builtin_def (STRSPN)
