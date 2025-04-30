/* strchr -- find the first instance of C in a nul-terminated string.
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

	.syntax unified
	.text

ENTRY (strchr)
	@ r0 = start of string
	@ r1 = character to match
	@ returns NULL for no match, or a pointer to the match
	ldrb	r2, [r0]		@ load the first byte asap
	uxtb	r1, r1

	@ To cater to long strings, we want to search through a few
	@ characters until we reach an aligned pointer.  To cater to
	@ small strings, we don't want to start doing word operations
	@ immediately.  The compromise is a maximum of 16 bytes less
	@ whatever is required to end with an aligned pointer.
	@ r3 = number of characters to search in alignment loop
	and	r3, r0, #7
	rsb	r3, r3, #15		@ 16 - 1 peeled loop iteration
	cmp	r2, r1			@ Found C?
	it	ne
	cmpne	r2, #0			@ Found EOS?
	beq	99f

	@ Loop until we find ...
1:	ldrb	r2, [r0, #1]!
	subs	r3, r3, #1		@ ... the aligment point
	it	ne
	cmpne	r2, r1			@ ... or the character
	it	ne
	cmpne	r2, #0			@ ... or EOS
	bne	1b

	@ Disambiguate the exit possibilites above
	cmp	r2, r1			@ Found the character
	it	ne
	cmpne	r2, #0			@ Found EOS
	beq	99f
	add	r0, r0, #1

	@ So now we're aligned.  Now we actually need a stack frame.
	push	{ r4, r5, r6, r7 }
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)

	ldrd	r2, r3, [r0], #8
	orr	r1, r1, r1, lsl #8	@ Replicate C to all bytes
#ifdef ARCH_HAS_T2
	movw	ip, #0x0101
	pld	[r0, #64]
	movt	ip, #0x0101
#else
	ldr	ip, =0x01010101
	pld	[r0, #64]
#endif
	orr	r1, r1, r1, lsl #16

	@ Loop searching for EOS or C, 8 bytes at a time.
2:
	@ Subtracting (unsigned saturating) from 1 means result of 1 for
	@ any byte that was originally zero and 0 otherwise.  Therefore
	@ we consider the lsb of each byte the "found" bit.
	uqsub8	r4, ip, r2		@ Find EOS
	eor	r6, r2, r1		@ Convert C bytes to 0
	uqsub8	r5, ip, r3
	eor	r7, r3, r1
	uqsub8	r6, ip, r6		@ Find C
	pld	[r0, #128]		@ Prefetch 2 lines ahead
	uqsub8	r7, ip, r7
	orr	r4, r4, r6		@ Combine found for EOS and C
	orr	r5, r5, r7
	orrs	r6, r4, r5		@ Combine the two words
	it	eq
	ldrdeq	r2, r3, [r0], #8
	beq	2b

	@ Found something.  Disambiguate between first and second words.
	@ Adjust r0 to point to the word containing the match.
	@ Adjust r2 to the contents of the word containing the match.
	@ Adjust r4 to the found bits for the word containing the match.
	cmp	r4, #0
	sub	r0, r0, #4
	itte	eq
	moveq	r4, r5
	moveq	r2, r3
	subne	r0, r0, #4

	@ Find the bit-offset of the match within the word.
#if defined(__ARMEL__)
	@ For LE, swap the found word so clz searches from the little end.
	rev	r4, r4
#else
	@ For BE, byte swap the word to make it easier to extract the byte.
	rev	r2, r2
#endif
	@ We're counting 0x01 (not 0x80), so the bit offset is 7 too high.
	clz	r3, r4
	sub	r3, r3, #7
	lsr	r2, r2, r3		@ Shift down found byte
	uxtb	r1, r1			@ Undo replication of C
	uxtb	r2, r2			@ Extract found byte
	add	r0, r0, r3, lsr #3	@ Adjust the pointer to the found byte

	pop	{ r4, r5, r6, r7 }
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)

	@ Disambiguate between EOS and C.
99:
	cmp	r2, r1
	it	ne
	movne	r0, #0			@ Found EOS, return NULL
	bx	lr

END (strchr)

weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
