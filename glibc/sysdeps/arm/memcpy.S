/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   Contributed by MontaVista Software, Inc. (written by Nicolas Pitre)

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Thumb requires excessive IT insns here.  */
#define NO_THUMB
#include <sysdep.h>
#include <arm-features.h>

/*
 * Data preload for architectures that support it (ARM V5TE and above)
 */
#if (!defined (__ARM_ARCH_2__) && !defined (__ARM_ARCH_3__) \
     && !defined (__ARM_ARCH_3M__) && !defined (__ARM_ARCH_4__) \
     && !defined (__ARM_ARCH_4T__) && !defined (__ARM_ARCH_5__) \
     && !defined (__ARM_ARCH_5T__))
#define PLD(code...)    code
#else
#define PLD(code...)
#endif

/*
 * This can be used to enable code to cacheline align the source pointer.
 * Experiments on tested architectures (StrongARM and XScale) didn't show
 * this a worthwhile thing to do.  That might be different in the future.
 */
//#define CALGN(code...)        code
#define CALGN(code...)

/*
 * Endian independent macros for shifting bytes within registers.
 */
#ifndef __ARMEB__
#define PULL            lsr
#define PUSH            lsl
#else
#define PULL            lsl
#define PUSH            lsr
#endif

		.text
		.syntax unified

/* Prototype: void *memcpy(void *dest, const void *src, size_t n); */

ENTRY(memcpy)

		push	{r0, r4, lr}
		cfi_adjust_cfa_offset (12)
		cfi_rel_offset (r4, 4)
		cfi_rel_offset (lr, 8)

		cfi_remember_state

		subs	r2, r2, #4
		blo	8f
		ands	ip, r0, #3
	PLD(	pld	[r1, #0]		)
		bne	9f
		ands	ip, r1, #3
		bne	10f

1:		subs	r2, r2, #(28)
		push	{r5 - r8}
		cfi_adjust_cfa_offset (16)
		cfi_rel_offset (r5, 0)
		cfi_rel_offset (r6, 4)
		cfi_rel_offset (r7, 8)
		cfi_rel_offset (r8, 12)
		blo	5f

	CALGN(	ands	ip, r1, #31		)
	CALGN(	rsb	r3, ip, #32		)
	CALGN(	sbcsne	r4, r3, r2		)  @ C is always set here
	CALGN(	bcs	2f			)
	CALGN(	adr	r4, 6f			)
	CALGN(	subs	r2, r2, r3		)  @ C gets set
#ifndef ARM_ALWAYS_BX
	CALGN(	add	pc, r4, ip, lsl	#(ARM_BX_ALIGN_LOG2 - 2))
#else
	CALGN(	add	r4, r4, ip, lsl	#(ARM_BX_ALIGN_LOG2 - 2))
	CALGN(	bx	r4			)
#endif

	PLD(	pld	[r1, #0]		)
2:	PLD(	cmp	r2, #96			)
	PLD(	pld	[r1, #28]		)
	PLD(	blo	4f			)
	PLD(	pld	[r1, #60]		)
	PLD(	pld	[r1, #92]		)

3:	PLD(	pld	[r1, #124]		)
4:		ldmia	r1!, {r3, r4, r5, r6, r7, r8, ip, lr}
		subs	r2, r2, #32
		stmia	r0!, {r3, r4, r5, r6, r7, r8, ip, lr}
		bhs	3b

5:		ands	ip, r2, #28
		rsb	ip, ip, #32
#ifndef ARM_ALWAYS_BX
		/* C is always clear here.  */
		addne	pc, pc, ip, lsl #(ARM_BX_ALIGN_LOG2 - 2)
		b	7f
#else
		beq	7f
		push	{r10}
		cfi_adjust_cfa_offset (4)
		cfi_rel_offset (r10, 0)
0:		add	r10, pc, ip, lsl #(ARM_BX_ALIGN_LOG2 - 2)
		/* If alignment is not perfect, then there will be some
		   padding (nop) instructions between this BX and label 6.
		   The computation above assumed that two instructions
		   later is exactly the right spot.  */
		add	r10, #(6f - (0b + PC_OFS))
		bx	r10
#endif
		.p2align ARM_BX_ALIGN_LOG2
6:		nop
		.p2align ARM_BX_ALIGN_LOG2
		ldr	r3, [r1], #4
		.p2align ARM_BX_ALIGN_LOG2
		ldr	r4, [r1], #4
		.p2align ARM_BX_ALIGN_LOG2
		ldr	r5, [r1], #4
		.p2align ARM_BX_ALIGN_LOG2
		ldr	r6, [r1], #4
		.p2align ARM_BX_ALIGN_LOG2
		ldr	r7, [r1], #4
		.p2align ARM_BX_ALIGN_LOG2
		ldr	r8, [r1], #4
		.p2align ARM_BX_ALIGN_LOG2
		ldr	lr, [r1], #4

#ifndef ARM_ALWAYS_BX
		add	pc, pc, ip, lsl #(ARM_BX_ALIGN_LOG2 - 2)
		nop
#else
0:		add	r10, pc, ip, lsl #(ARM_BX_ALIGN_LOG2 - 2)
		/* If alignment is not perfect, then there will be some
		   padding (nop) instructions between this BX and label 66.
		   The computation above assumed that two instructions
		   later is exactly the right spot.  */
		add	r10, #(66f - (0b + PC_OFS))
		bx	r10
#endif
		.p2align ARM_BX_ALIGN_LOG2
66:		nop
		.p2align ARM_BX_ALIGN_LOG2
		str	r3, [r0], #4
		.p2align ARM_BX_ALIGN_LOG2
		str	r4, [r0], #4
		.p2align ARM_BX_ALIGN_LOG2
		str	r5, [r0], #4
		.p2align ARM_BX_ALIGN_LOG2
		str	r6, [r0], #4
		.p2align ARM_BX_ALIGN_LOG2
		str	r7, [r0], #4
		.p2align ARM_BX_ALIGN_LOG2
		str	r8, [r0], #4
		.p2align ARM_BX_ALIGN_LOG2
		str	lr, [r0], #4

#ifdef ARM_ALWAYS_BX
		pop	{r10}
		cfi_adjust_cfa_offset (-4)
		cfi_restore (r10)
#endif

	CALGN(	bcs	2b			)

7:		pop	{r5 - r8}
		cfi_adjust_cfa_offset (-16)
		cfi_restore (r5)
		cfi_restore (r6)
		cfi_restore (r7)
		cfi_restore (r8)

8:		movs	r2, r2, lsl #31
		ldrbne	r3, [r1], #1
		ldrbcs	r4, [r1], #1
		ldrbcs	ip, [r1]
		strbne	r3, [r0], #1
		strbcs	r4, [r0], #1
		strbcs	ip, [r0]

#if ((defined (__ARM_ARCH_4T__) && defined(__THUMB_INTERWORK__)) \
     || defined (ARM_ALWAYS_BX))
		pop	{r0, r4, lr}
		cfi_adjust_cfa_offset (-12)
		cfi_restore (r4)
		cfi_restore (lr)
		bx      lr
#else
		pop	{r0, r4, pc}
#endif

		cfi_restore_state

9:		rsb	ip, ip, #4
		cmp	ip, #2
		ldrbgt	r3, [r1], #1
		ldrbge	r4, [r1], #1
		ldrb	lr, [r1], #1
		strbgt	r3, [r0], #1
		strbge	r4, [r0], #1
		subs	r2, r2, ip
		strb	lr, [r0], #1
		blo	8b
		ands	ip, r1, #3
		beq	1b

10:		bic	r1, r1, #3
		cmp	ip, #2
		ldr	lr, [r1], #4
		beq	17f
		bgt	18f


		.macro	forward_copy_shift pull push

		subs	r2, r2, #28
		blo	14f

	CALGN(	ands	ip, r1, #31		)
	CALGN(	rsb	ip, ip, #32		)
	CALGN(	sbcsne	r4, ip, r2		)  @ C is always set here
	CALGN(	subcc	r2, r2, ip		)
	CALGN(	bcc	15f			)

11:		push	{r5 - r8, r10}
		cfi_adjust_cfa_offset (20)
		cfi_rel_offset (r5, 0)
		cfi_rel_offset (r6, 4)
		cfi_rel_offset (r7, 8)
		cfi_rel_offset (r8, 12)
		cfi_rel_offset (r10, 16)

	PLD(	pld	[r1, #0]		)
	PLD(	cmp	r2, #96			)
	PLD(	pld	[r1, #28]		)
	PLD(	blo	13f			)
	PLD(	pld	[r1, #60]		)
	PLD(	pld	[r1, #92]		)

12:	PLD(	pld	[r1, #124]		)
13:		ldmia	r1!, {r4, r5, r6, r7}
		mov	r3, lr, PULL #\pull
		subs	r2, r2, #32
		ldmia	r1!, {r8, r10, ip, lr}
		orr	r3, r3, r4, PUSH #\push
		mov	r4, r4, PULL #\pull
		orr	r4, r4, r5, PUSH #\push
		mov	r5, r5, PULL #\pull
		orr	r5, r5, r6, PUSH #\push
		mov	r6, r6, PULL #\pull
		orr	r6, r6, r7, PUSH #\push
		mov	r7, r7, PULL #\pull
		orr	r7, r7, r8, PUSH #\push
		mov	r8, r8, PULL #\pull
		orr	r8, r8, r10, PUSH #\push
		mov	r10, r10, PULL #\pull
		orr	r10, r10, ip, PUSH #\push
		mov	ip, ip, PULL #\pull
		orr	ip, ip, lr, PUSH #\push
		stmia	r0!, {r3, r4, r5, r6, r7, r8, r10, ip}
		bhs	12b

		pop	{r5 - r8, r10}
		cfi_adjust_cfa_offset (-20)
		cfi_restore (r5)
		cfi_restore (r6)
		cfi_restore (r7)
		cfi_restore (r8)
		cfi_restore (r10)

14:		ands	ip, r2, #28
		beq	16f

15:		mov	r3, lr, PULL #\pull
		ldr	lr, [r1], #4
		subs	ip, ip, #4
		orr	r3, r3, lr, PUSH #\push
		str	r3, [r0], #4
		bgt	15b
	CALGN(	cmp	r2, #0			)
	CALGN(	bge	11b			)

16:		sub	r1, r1, #(\push / 8)
		b	8b

		.endm


		forward_copy_shift	pull=8	push=24

17:		forward_copy_shift	pull=16	push=16

18:		forward_copy_shift	pull=24	push=8

END(memcpy)
libc_hidden_builtin_def (memcpy)
