/* strrchr -- find the last occurence of C in a nul-terminated string
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

ENTRY (strrchr)
	@ r0 = start of string
	@ r1 = character to match
	@ returns NULL for no match, or a pointer to the match

	mov	r3, r0
	mov	r0, #0
	uxtb	r1, r1

	@ Loop a few times until we're aligned.
	tst	r3, #7
	beq	2f
1:	ldrb	r2, [r3], #1
	cmp	r2, r1			@ Find the character
	it	eq
	subeq	r0, r3, #1
	cmp	r2, #0			@ Find EOS
	it	eq
	bxeq	lr
	tst	r3, #7			@ Find the aligment point
	bne	1b

	@ So now we're aligned.  Now we actually need a stack frame.
2:	push	{ r4, r5, r6, r7 }
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)

	orr	r1, r1, r1, lsl #8	@ Replicate C to all bytes
#ifdef ARCH_HAS_T2
	movw	ip, #0x0101
	movt	ip, #0x0101
#else
	ldr	ip, =0x01010101
#endif
	orr	r1, r1, r1, lsl #16
	mov	r2, #0			@ No found bits yet

	@ Loop searching for EOS and C, 8 bytes at a time.
	@ Any time we find a match in a word, we copy the address of
	@ the word to r0, and the found bits to r2.
3:	ldrd	r4, r5, [r3], #8
	@ Subtracting (unsigned saturating) from 1 means result of 1 for
	@ any byte that was originally zero and 0 otherwise.  Therefore
	@ we consider the lsb of each byte the "found" bit.
	uqsub8	r6, ip, r4		@ Find EOS
	uqsub8	r7, ip, r5
	eor	r4, r4, r1		@ Convert C bytes to 0
	eor	r5, r5, r1
	uqsub8	r4, ip, r4		@ Find C
	uqsub8	r5, ip, r5
	cmp	r6, #0			@ Found EOS, first word
	bne	4f
	cmp	r4, #0			@ Handle C, first word
	itt	ne
	subne	r0, r3, #8
	movne	r2, r4
	cmp	r7, #0			@ Found EOS, second word
	bne	5f
	cmp	r5, #0			@ Handle C, second word
	itt	ne
	subne	r0, r3, #4
	movne	r2, r5
	b	3b

	@ Found EOS in second word; fold to first word.
5:	add	r3, r3, #4		@ Dec pointer to 2nd word, with below
	mov	r4, r5			@ Overwrite first word C found
	mov	r6, r7			@ Overwrite first word EOS found

	@ Found EOS.  Zap found C after EOS.
4:	sub	r3, r3, #8		@ Decrement pointer to first word
#ifdef __ARMEB__
	@ Byte swap to be congruent with LE, which is easier from here on.
	rev	r6, r6			@ Byte swap found EOS,
	rev	r4, r4			@ ... this found C
	rev	r2, r2			@ ... prev found C
#endif
	sub	r7, r6, #1		@ Toggle EOS lsb and below
	eor	r6, r6, r7		@ All bits below and including lsb
	ands	r4, r4, r6		@ Zap C above EOS
	itt	ne
	movne	r2, r4			@ Copy to result, if still non-zero
	movne	r0, r3

	pop	{ r4, r5, r6, r7 }
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)

	@ Adjust the result pointer if we found a word containing C.
	cmp	r2, #0
	clz	r2, r2			@ Find the bit offset of the last C
	itt	ne
	rsbne	r2, r2, #32		@ Convert to a count from the right
	addne	r0, r0, r2, lsr #3	@ Convert to byte offset and add.
	bx	lr

END (strrchr)

weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
