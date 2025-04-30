/* strcpy -- copy a nul-terminated string.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* Endian independent macros for shifting bytes within registers.  */
#ifdef __ARMEB__
#define lsh_gt		lsr
#define lsh_ls		lsl
#else
#define lsh_gt		lsl
#define lsh_ls		lsr
#endif

	.syntax unified
	.text

ENTRY (__stpcpy)
	@ Signal stpcpy with NULL in IP.
	mov	ip, #0
	b	0f
END (__stpcpy)

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)

ENTRY (strcpy)
	@ Signal strcpy with DEST in IP.
	mov	ip, r0
0:
	pld	[r0, #0]
	pld	[r1, #0]

	@ To cater to long strings, we want 8 byte alignment in the source.
	@ To cater to small strings, we don't want to start that right away.
	@ Loop up to 16 times, less whatever it takes to reach alignment.
	and	r3, r1, #7
	rsb	r3, r3, #16

	@ Loop until we find ...
1:	ldrb	r2, [r1], #1
	subs	r3, r3, #1		@ ... the alignment point
	strb	r2, [r0], #1
	it	ne
	cmpne	r2, #0			@ ... or EOS
	bne	1b

	@ Disambiguate the exit possibilites above
	cmp	r2, #0			@ Found EOS
	beq	.Lreturn

	@ Load the next two words asap
	ldrd	r2, r3, [r1], #8
	pld	[r0, #64]
	pld	[r1, #64]

	@ For longer strings, we actaully need a stack frame.
	push	{ r4, r5, r6, r7 }
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)

	@ Subtracting (unsigned saturating) from 1 for any byte means result
	@ of 1 for any byte that was originally zero and 0 otherwise.
	@ Therefore we consider the lsb of each byte the "found" bit.
#ifdef ARCH_HAS_T2
	movw	r7, #0x0101
	tst	r0, #3			@ Test alignment of DEST
	movt	r7, #0x0101
#else
	ldr	r7, =0x01010101
	tst	r0, #3
#endif
	bne	.Lunaligned

	@ So now source (r1) is aligned to 8, and dest (r0) is aligned to 4.
	@ Loop, reading 8 bytes at a time, searching for EOS.
	.balign	16
2:	uqsub8	r4, r7, r2		@ Find EOS
	uqsub8	r5, r7, r3
	pld	[r1, #128]
	cmp	r4, #0			@ EOS in first word?
	pld	[r0, #128]
	bne	3f
	str	r2, [r0], #4
	cmp	r5, #0			@ EOS in second word?
	bne	4f
	str	r3, [r0], #4
	ldrd	r2, r3, [r1], #8
	b	2b

3:	sub	r1, r1, #4		@ backup to first word
4:	sub	r1, r1, #4		@ backup to second word

	@ ... then finish up any tail a byte at a time.
	@ Note that we generally back up and re-read source bytes,
	@ but we'll not re-write dest bytes.
.Lbyte_loop:
	ldrb	r2, [r1], #1
	cmp	r2, #0
	strb	r2, [r0], #1
	bne	.Lbyte_loop

	pop	{ r4, r5, r6, r7 }
	cfi_remember_state
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)

.Lreturn:
	cmp	ip, #0			@ Was this strcpy or stpcpy?
	ite	eq
	subeq	r0, r0, #1		@ stpcpy: undo post-inc from store
	movne	r0, ip			@ strcpy: return original dest
	bx	lr

.Lunaligned:
	cfi_restore_state
	@ Here, source is aligned to 8, but the destination is not word
	@ aligned.  Therefore we have to shift the data in order to be
	@ able to perform aligned word stores.

	@ Find out which misalignment we're dealing with.
	tst	r0, #1
	beq	.Lunaligned2
	tst	r0, #2
	bne	.Lunaligned3
	@ Fallthru to .Lunaligned1.

.macro unaligned_copy	unalign
	@ Prologue to unaligned loop.  Seed shifted non-zero bytes.
	uqsub8	r4, r7, r2		@ Find EOS
	uqsub8	r5, r7, r3
	cmp	r4, #0			@ EOS in first word?
	it	ne
	subne	r1, r1, #8
	bne	.Lbyte_loop
#ifdef __ARMEB__
	rev	r2, r2			@ Byte stores below need LE data
#endif
	@ Store a few bytes from the first word.
	@ At the same time we align r0 and shift out bytes from r2.
.rept	4-\unalign
	strb	r2, [r0], #1
	lsr	r2, r2, #8
.endr
#ifdef __ARMEB__
	rev	r2, r2			@ Undo previous rev
#endif
	@ Rotated unaligned copy loop.  The tail of the prologue is
	@ shared with the loop itself.
	.balign 8
1:	cmp	r5, #0			@ EOS in second word?
	bne	4f
	@ Combine first and second words
	orr	r2, r2, r3, lsh_gt #(\unalign*8)
	@ Save leftover bytes from the two words
	lsh_ls	r6, r3, #((4-\unalign)*8)
	str	r2, [r0], #4
	@ The "real" start of the unaligned copy loop.
	ldrd	r2, r3, [r1], #8	@ Load 8 more bytes
	uqsub8	r4, r7, r2		@ Find EOS
	pld	[r1, #128]
	uqsub8	r5, r7, r3
	pld	[r0, #128]
	cmp	r4, #0			@ EOS in first word?
	bne	3f
	@ Combine the leftover and the first word
	orr	r6, r6, r2, lsh_gt #(\unalign*8)
	@ Discard used bytes from the first word.
	lsh_ls	r2, r2, #((4-\unalign)*8)
	str	r6, [r0], #4
	b	1b
	@ Found EOS in one of the words; adjust backward
3:	sub	r1, r1, #4
	mov	r2, r6
4:	sub	r1, r1, #4
	@ And store the remaining bytes from the leftover
#ifdef __ARMEB__
	rev	r2, r2
#endif
.rept	\unalign
	strb	r2, [r0], #1
	lsr	r2, r2, #8
.endr
	b	.Lbyte_loop
.endm

.Lunaligned1:
	unaligned_copy	1
.Lunaligned2:
	unaligned_copy	2
.Lunaligned3:
	unaligned_copy	3

END (strcpy)

libc_hidden_builtin_def (strcpy)
