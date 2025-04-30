/* Optimized strlen implementation for PowerPC64/POWER9.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef STRLEN
# define STRLEN __strlen
# define DEFINE_STRLEN_HIDDEN_DEF 1
#endif

/* Implements the function

   int [r3] strlen (const void *s [r3])

   The implementation can load bytes past a matching byte, but only
   up to the next 64B boundary, so it never crosses a page.  */

.machine power9
ENTRY_TOCLESS (STRLEN, 4)
	CALL_MCOUNT 2

	vspltisb  v18,0
	vspltisb  v19,-1

	neg	  r5,r3
	rldicl	  r9,r5,0,60   /* How many bytes to get source 16B aligned?  */

	/* Align data and fill bytes not loaded with non matching char.  */
	lvx	  v0,0,r3
	lvsr	  v1,0,r3
	vperm	  v0,v19,v0,v1

	vcmpequb. v6,v0,v18
	beq	  cr6,L(aligned)

	vctzlsbb  r3,v6
	blr

	/* Test 64B 16B at a time.  The 64B vector loop is optimized for
	   longer strings.  Likewise, we check a multiple of 64B to avoid
	   breaking the alignment calculation below.  */
L(aligned):
	add	  r4,r3,r9
	rldicl.	  r5,r4,60,62  /* Determine the number of 48B loops needed for
                                  alignment to 64B.  And test for zero.  */

	lxv	  v0+32,0(r4)
	vcmpequb. v6,v0,v18
	bne	  cr6,L(tail1)

	lxv	  v0+32,16(r4)
	vcmpequb. v6,v0,v18
	bne 	  cr6,L(tail2)

	lxv	  v0+32,32(r4)
	vcmpequb. v6,v0,v18
	bne 	  cr6,L(tail3)

	lxv	  v0+32,48(r4)
	vcmpequb. v6,v0,v18
	bne 	  cr6,L(tail4)
	addi	  r4,r4,64

	/* Speculatively generate a fake 16B aligned address to generate the
	   vector byte constant 0,1,..,15 using lvsl during reduction.  */
	li	  r0,0

	/* Skip the alignment if already 64B aligned.  */
	beq	  L(loop_64b)
	mtctr	  r5

	/* Test 48B per iteration until 64B aligned.  */
	.p2align  5
L(loop):
	lxv	  v0+32,0(r4)
	vcmpequb. v6,v0,v18
	bne	  cr6,L(tail1)

	lxv	  v0+32,16(r4)
	vcmpequb. v6,v0,v18
	bne	  cr6,L(tail2)

	lxv 	  v0+32,32(r4)
	vcmpequb. v6,v0,v18
	bne	  cr6,L(tail3)

	addi	  r4,r4,48
	bdnz	  L(loop)

	.p2align  5
L(loop_64b):
	lxv	  v1+32,0(r4)     /* Load 4 quadwords.  */
	lxv	  v2+32,16(r4)
	lxv	  v3+32,32(r4)
	lxv	  v4+32,48(r4)
	vminub	  v5,v1,v2        /* Compare and merge into one VR for speed.  */
	vminub	  v6,v3,v4
	vminub	  v7,v5,v6
	vcmpequb. v7,v7,v18       /* Check for NULLs.  */
	addi	  r4,r4,64        /* Adjust address for the next iteration.  */
	bne	  cr6,L(vmx_zero)

	lxv	  v1+32,0(r4)     /* Load 4 quadwords.  */
	lxv	  v2+32,16(r4)
	lxv	  v3+32,32(r4)
	lxv	  v4+32,48(r4)
	vminub	  v5,v1,v2        /* Compare and merge into one VR for speed.  */
	vminub	  v6,v3,v4
	vminub	  v7,v5,v6
	vcmpequb. v7,v7,v18       /* Check for NULLs.  */
	addi	  r4,r4,64        /* Adjust address for the next iteration.  */
	bne	  cr6,L(vmx_zero)

	lxv	  v1+32,0(r4)     /* Load 4 quadwords.  */
	lxv	  v2+32,16(r4)
	lxv	  v3+32,32(r4)
	lxv	  v4+32,48(r4)
	vminub	  v5,v1,v2        /* Compare and merge into one VR for speed.  */
	vminub	  v6,v3,v4
	vminub	  v7,v5,v6
	vcmpequb. v7,v7,v18       /* Check for NULLs.  */
	addi	  r4,r4,64        /* Adjust address for the next iteration.  */
	beq	  cr6,L(loop_64b)

L(vmx_zero):
	/* OK, we found a null byte.  Let's look for it in the current 64-byte
	   block and mark it in its corresponding VR.  */
	vcmpequb  v1,v1,v18
	vcmpequb  v2,v2,v18
	vcmpequb  v3,v3,v18
	vcmpequb  v4,v4,v18

	/* We will now 'compress' the result into a single doubleword, so it
	   can be moved to a GPR for the final calculation.  First, we
	   generate an appropriate mask for vbpermq, so we can permute bits into
	   the first halfword.  */
	vspltisb  v10,3
	lvsl	  v11,0,r0
	vslb	  v10,v11,v10

	/* Permute the first bit of each byte into bits 48-63.  */
	vbpermq	  v1,v1,v10
	vbpermq	  v2,v2,v10
	vbpermq	  v3,v3,v10
	vbpermq	  v4,v4,v10

	/* Shift each component into its correct position for merging.  */
	vsldoi	  v2,v2,v2,2
	vsldoi	  v3,v3,v3,4
	vsldoi	  v4,v4,v4,6

	/* Merge the results and move to a GPR.  */
	vor	  v1,v2,v1
	vor	  v2,v3,v4
	vor	  v4,v1,v2
	mfvrd	  r10,v4

	/* Adjust address to the begninning of the current 64-byte block.  */
	addi	  r4,r4,-64

	cnttzd	  r0,r10           /* Count trailing zeros before the match.  */
	subf	  r5,r3,r4
	add	  r3,r5,r0         /* Compute final length.  */
	blr

L(tail1):
	vctzlsbb  r0,v6
	add	  r4,r4,r0
	subf	  r3,r3,r4
	blr

L(tail2):
	vctzlsbb  r0,v6
	add	  r4,r4,r0
	addi	  r4,r4,16
	subf	  r3,r3,r4
	blr

L(tail3):
	vctzlsbb  r0,v6
	add	  r4,r4,r0
	addi	  r4,r4,32
	subf	  r3,r3,r4
	blr

L(tail4):
	vctzlsbb  r0,v6
	add	  r4,r4,r0
	addi	  r4,r4,48
	subf	  r3,r3,r4
	blr

END (STRLEN)

#ifdef DEFINE_STRLEN_HIDDEN_DEF
weak_alias (__strlen, strlen)
libc_hidden_builtin_def (strlen)
#endif
