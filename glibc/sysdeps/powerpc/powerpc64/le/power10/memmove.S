/* Optimized memmove implementation for POWER10.
   Copyright (C) 2021 Free Software Foundation, Inc.
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


/* void* [r3] memmove (void *dest [r3], const void *src [r4], size_t len [r5])

   This optimization checks if 'src' and 'dst' overlap.  If they do not
   or 'src' is ahead of 'dest' then it copies forward.
   Otherwise, an optimized backward copy is used.  */

#ifndef MEMMOVE
# define MEMMOVE memmove
#endif
	.machine power9
ENTRY_TOCLESS (MEMMOVE, 5)
	CALL_MCOUNT 3

L(_memmove):
	.p2align 5
	/* Check if there is overlap, if so it will branch to backward copy.  */
	subf	r9,r4,r3
	cmpld	cr7,r9,r5
	blt	cr7,L(memmove_bwd)

	/* Fast path for length shorter than 16 bytes.  */
	sldi	r7,r5,56
	lxvl	32+v2,r4,r7
	stxvl	32+v2,r3,r7
	subic.	r8,r5,16
	blelr

	/* For shorter lengths aligning the dest address to 16 bytes either
	   decreases performance or is irrelevant.  I'm making use of this
	   comparison to skip the alignment in.  */
	cmpldi	cr6,r5,256
	bge	cr6,L(ge_256)
	/* Account for the first 16-byte copy.  */
	addi	r4,r4,16
	addi	r11,r3,16	/* use r11 to keep dest address on r3.  */
	subi	r5,r5,16
	b	L(loop_head)

	.p2align 5
L(ge_256):
	/* Account for the first copy <= 16 bytes.  This is necessary for
	   memmove because at this point the src address can be in front of the
	   dest address.  */
	clrldi	r9,r5,56
	li	r8,16
	cmpldi	r9,16
	iselgt	r9,r8,r9
	add	r4,r4,r9
	add	r11,r3,r9	/* use r11 to keep dest address on r3.  */
	sub	r5,r5,r9

	/* Align dest to 16 bytes.  */
	neg	r7,r3
	clrldi.	r9,r7,60
	beq	L(loop_head)

	.p2align 5
	sldi	r6,r9,56
	lxvl	32+v0,r4,r6
	stxvl	32+v0,r11,r6
	sub	r5,r5,r9
	add	r4,r4,r9
	add	r11,r11,r9

L(loop_head):
	cmpldi	r5,63
	ble	L(final_64)

	srdi.	r7,r5,7
	beq	L(loop_tail)

	mtctr	r7

/* Main loop that copies 128 bytes each iteration.  */
	.p2align 5
L(loop):
	addi	r9,r4,64
	addi	r10,r11,64

	lxv	32+v0,0(r4)
	lxv	32+v1,16(r4)
	lxv	32+v2,32(r4)
	lxv	32+v3,48(r4)

	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)

	addi	r4,r4,128
	addi	r11,r11,128

	lxv	32+v4,0(r9)
	lxv	32+v5,16(r9)
	lxv	32+v6,32(r9)
	lxv	32+v7,48(r9)

	stxv	32+v4,0(r10)
	stxv	32+v5,16(r10)
	stxv	32+v6,32(r10)
	stxv	32+v7,48(r10)

	bdnz	L(loop)
	clrldi.	r5,r5,57
	beqlr

/* Copy 64 bytes.  */
	.p2align 5
L(loop_tail):
	cmpldi 	cr5,r5,63
	ble	cr5,L(final_64)

	lxv	32+v0,0(r4)
	lxv	32+v1,16(r4)
	lxv	32+v2,32(r4)
	lxv	32+v3,48(r4)

	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)

	addi	r4,r4,64
	addi	r11,r11,64
	subi	r5,r5,64

/* Copies the last 1-63 bytes.  */
	.p2align 5
L(final_64):
	/* r8 holds the number of bytes that will be copied with lxv/stxv.  */
	clrrdi.	r8,r5,4
	beq	L(tail1)

	cmpldi  cr5,r5,32
	lxv	32+v0,0(r4)
	blt	cr5,L(tail2)

	cmpldi	cr6,r5,48
	lxv	32+v1,16(r4)
	blt	cr6,L(tail3)

	.p2align 5
	lxv	32+v2,32(r4)
	stxv	32+v2,32(r11)
L(tail3):
	stxv	32+v1,16(r11)
L(tail2):
	stxv	32+v0,0(r11)
	sub	r5,r5,r8
	add	r4,r4,r8
	add	r11,r11,r8
	.p2align 5
L(tail1):
	sldi	r6,r5,56
	lxvl	v4,r4,r6
	stxvl	v4,r11,r6
	blr

/* If dest and src overlap, we should copy backwards.  */
L(memmove_bwd):
	add	r11,r3,r5
	add	r4,r4,r5

	/* Optimization for length smaller than 16 bytes.  */
	cmpldi	cr5,r5,15
	ble	cr5,L(tail1_bwd)

	/* For shorter lengths the alignment either slows down or is irrelevant.
	   The forward copy uses a already need 256 comparison for that.  Here
	   it's using 128 as it will reduce code and improve readability.  */
	cmpldi	cr7,r5,128
	blt	cr7,L(bwd_loop_tail)

	/* Align dest address to 16 bytes.  */
	.p2align 5
	clrldi.	r9,r11,60
	beq	L(bwd_loop_head)
	sub	r4,r4,r9
	sub	r11,r11,r9
	lxv	32+v0,0(r4)
	sldi	r6,r9,56
	stxvl   32+v0,r11,r6
	sub	r5,r5,r9

L(bwd_loop_head):
	srdi.	r7,r5,7
	beq	L(bwd_loop_tail)

	mtctr	r7

/* Main loop that copies 128 bytes every iteration.  */
	.p2align 5
L(bwd_loop):
	addi	r9,r4,-64
	addi	r10,r11,-64

	lxv	32+v0,-16(r4)
	lxv	32+v1,-32(r4)
	lxv	32+v2,-48(r4)
	lxv	32+v3,-64(r4)

	stxv	32+v0,-16(r11)
	stxv	32+v1,-32(r11)
	stxv	32+v2,-48(r11)
	stxv	32+v3,-64(r11)

	addi	r4,r4,-128
	addi	r11,r11,-128

	lxv	32+v0,-16(r9)
	lxv	32+v1,-32(r9)
	lxv	32+v2,-48(r9)
	lxv	32+v3,-64(r9)

	stxv	32+v0,-16(r10)
	stxv	32+v1,-32(r10)
	stxv	32+v2,-48(r10)
	stxv	32+v3,-64(r10)

	bdnz	L(bwd_loop)
	clrldi.	r5,r5,57
	beqlr

/* Copy 64 bytes.  */
	.p2align 5
L(bwd_loop_tail):
	cmpldi 	cr5,r5,63
	ble	cr5,L(bwd_final_64)

	addi	r4,r4,-64
	addi	r11,r11,-64

	lxv	32+v0,0(r4)
	lxv	32+v1,16(r4)
	lxv	32+v2,32(r4)
	lxv	32+v3,48(r4)

	stxv	32+v0,0(r11)
	stxv	32+v1,16(r11)
	stxv	32+v2,32(r11)
	stxv	32+v3,48(r11)

	subi	r5,r5,64

/* Copies the last 1-63 bytes.  */
	.p2align 5
L(bwd_final_64):
	/* r8 holds the number of bytes that will be copied with lxv/stxv.  */
	clrrdi.	r8,r5,4
	beq	L(tail1_bwd)

	cmpldi	cr5,r5,32
	lxv	32+v2,-16(r4)
	blt	cr5,L(tail2_bwd)

	cmpldi	cr6,r5,48
	lxv	32+v1,-32(r4)
	blt	cr6,L(tail3_bwd)

	.p2align 5
	lxv	32+v0,-48(r4)
	stxv	32+v0,-48(r11)
L(tail3_bwd):
	stxv	32+v1,-32(r11)
L(tail2_bwd):
	stxv	32+v2,-16(r11)
	sub	r4,r4,r5
	sub	r11,r11,r5
	sub	r5,r5,r8
	sldi	r6,r5,56
	lxvl	v4,r4,r6
	stxvl	v4,r11,r6
	blr

/* Copy last 16 bytes.  */
	.p2align 5
L(tail1_bwd):
	sub	r4,r4,r5
	sub	r11,r11,r5
	sldi	r6,r5,56
	lxvl	v4,r4,r6
	stxvl	v4,r11,r6
	blr

END_GEN_TB (MEMMOVE,TB_TOCLESS)
libc_hidden_builtin_def (memmove)

/* void bcopy(const void *src [r3], void *dest [r4], size_t n [r5])
   Implemented in this file to avoid linker create a stub function call
   in the branch to '_memmove'.  */
ENTRY_TOCLESS (__bcopy)
	mr	r6,r3
	mr	r3,r4
	mr	r4,r6
	b	L(_memmove)
END (__bcopy)
#ifndef __bcopy
weak_alias (__bcopy, bcopy)
#endif
