/* strlen -- find the length of a nul-terminated string.
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

ENTRY (strlen)
	@ r0 = start of string
	ldrb	r2, [r0]		@ load the first byte asap

	@ To cater to long strings, we want to search through a few
	@ characters until we reach an aligned pointer.  To cater to
	@ small strings, we don't want to start doing word operations
	@ immediately.  The compromise is a maximum of 16 bytes less
	@ whatever is required to end with an aligned pointer.
	@ r3 = number of characters to search in alignment loop
	and	r3, r0, #7
	mov	r1, r0			@ Save the input pointer
	rsb	r3, r3, #15		@ 16 - 1 peeled loop iteration
	cmp	r2, #0
	beq	99f

	@ Loop until we find ...
1:	ldrb	r2, [r0, #1]!
	subs	r3, r3, #1		@ ... the aligment point
	it	ne
	cmpne	r2, #0			@ ... or EOS
	bne	1b

	@ Disambiguate the exit possibilites above
	cmp	r2, #0			@ Found EOS
	beq	99f
	add	r0, r0, #1

	@ So now we're aligned.
	ldrd	r2, r3, [r0], #8
#ifdef ARCH_HAS_T2
	movw	ip, #0x0101
	pld	[r0, #64]
	movt	ip, #0x0101
#else
	ldr	ip, =0x01010101
	pld	[r0, #64]
#endif

	@ Loop searching for EOS, 8 bytes at a time.
	@ Subtracting (unsigned saturating) from 1 for any byte means that
	@ we get 1 for any byte that was originally zero and 0 otherwise.
	@ Therefore we consider the lsb of each byte the "found" bit.
	.balign	16
2:	uqsub8	r2, ip, r2		@ Find EOS
	uqsub8	r3, ip, r3
	pld	[r0, #128]		@ Prefetch 2 lines ahead
	orrs	r3, r3, r2		@ Combine the two words
	it	eq
	ldrdeq	r2, r3, [r0], #8
	beq	2b

	@ Found something.  Disambiguate between first and second words.
	@ Adjust r0 to point to the word containing the match.
	@ Adjust r2 to the found bits for the word containing the match.
	cmp	r2, #0
	sub	r0, r0, #4
	ite	eq
	moveq	r2, r3
	subne	r0, r0, #4

	@ Find the bit-offset of the match within the word.  Note that the
	@ bit result from clz will be 7 higher than "true", but we'll
	@ immediately discard those bits converting to a byte offset.
#ifdef __ARMEL__
	rev	r2, r2			@ For LE, count from the little end
#endif
	clz	r2, r2
	add	r0, r0, r2, lsr #3	@ Adjust the pointer to the found byte
99:
	sub	r0, r0, r1		@ Subtract input to compute length
	bx	lr

END (strlen)

libc_hidden_builtin_def (strlen)
