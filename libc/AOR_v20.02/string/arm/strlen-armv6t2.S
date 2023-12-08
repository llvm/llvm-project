/*
 * strlen - calculate the length of a string
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#if __ARM_ARCH >= 6 && __ARM_ARCH_ISA_THUMB == 2

/*
   Assumes:
   ARMv6T2, AArch32

 */

#include "../asmdefs.h"

#ifdef __ARMEB__
#define S2LO		lsl
#define S2HI		lsr
#else
#define S2LO		lsr
#define S2HI		lsl
#endif

	/* This code requires Thumb.  */
	.thumb
	.syntax unified

/* Parameters and result.  */
#define srcin		r0
#define result		r0

/* Internal variables.  */
#define src		r1
#define data1a		r2
#define data1b		r3
#define const_m1	r12
#define const_0		r4
#define tmp1		r4		/* Overlaps const_0  */
#define tmp2		r5

ENTRY (__strlen_armv6t2)
	pld	[srcin, #0]
	strd	r4, r5, [sp, #-8]!
	bic	src, srcin, #7
	mvn	const_m1, #0
	ands	tmp1, srcin, #7		/* (8 - bytes) to alignment.  */
	pld	[src, #32]
	bne.w	L(misaligned8)
	mov	const_0, #0
	mov	result, #-8
L(loop_aligned):
	/* Bytes 0-7.  */
	ldrd	data1a, data1b, [src]
	pld	[src, #64]
	add	result, result, #8
L(start_realigned):
	uadd8	data1a, data1a, const_m1	/* Saturating GE<0:3> set.  */
	sel	data1a, const_0, const_m1	/* Select based on GE<0:3>.  */
	uadd8	data1b, data1b, const_m1
	sel	data1b, data1a, const_m1	/* Only used if d1a == 0.  */
	cbnz	data1b, L(null_found)

	/* Bytes 8-15.  */
	ldrd	data1a, data1b, [src, #8]
	uadd8	data1a, data1a, const_m1	/* Saturating GE<0:3> set.  */
	add	result, result, #8
	sel	data1a, const_0, const_m1	/* Select based on GE<0:3>.  */
	uadd8	data1b, data1b, const_m1
	sel	data1b, data1a, const_m1	/* Only used if d1a == 0.  */
	cbnz	data1b, L(null_found)

	/* Bytes 16-23.  */
	ldrd	data1a, data1b, [src, #16]
	uadd8	data1a, data1a, const_m1	/* Saturating GE<0:3> set.  */
	add	result, result, #8
	sel	data1a, const_0, const_m1	/* Select based on GE<0:3>.  */
	uadd8	data1b, data1b, const_m1
	sel	data1b, data1a, const_m1	/* Only used if d1a == 0.  */
	cbnz	data1b, L(null_found)

	/* Bytes 24-31.  */
	ldrd	data1a, data1b, [src, #24]
	add	src, src, #32
	uadd8	data1a, data1a, const_m1	/* Saturating GE<0:3> set.  */
	add	result, result, #8
	sel	data1a, const_0, const_m1	/* Select based on GE<0:3>.  */
	uadd8	data1b, data1b, const_m1
	sel	data1b, data1a, const_m1	/* Only used if d1a == 0.  */
	cmp	data1b, #0
	beq	L(loop_aligned)

L(null_found):
	cmp	data1a, #0
	itt	eq
	addeq	result, result, #4
	moveq	data1a, data1b
#ifndef __ARMEB__
	rev	data1a, data1a
#endif
	clz	data1a, data1a
	ldrd	r4, r5, [sp], #8
	add	result, result, data1a, lsr #3	/* Bits -> Bytes.  */
	bx	lr

L(misaligned8):
	ldrd	data1a, data1b, [src]
	and	tmp2, tmp1, #3
	rsb	result, tmp1, #0
	lsl	tmp2, tmp2, #3			/* Bytes -> bits.  */
	tst	tmp1, #4
	pld	[src, #64]
	S2HI	tmp2, const_m1, tmp2
	orn	data1a, data1a, tmp2
	itt	ne
	ornne	data1b, data1b, tmp2
	movne	data1a, const_m1
	mov	const_0, #0
	b	L(start_realigned)

END (__strlen_armv6t2)

#endif /* __ARM_ARCH >= 6 && __ARM_ARCH_ISA_THUMB == 2  */
