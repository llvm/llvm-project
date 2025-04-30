/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Code contributed by Dave Gilbert <david.gilbert@linaro.org>

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

@ This memchr routine is optimised on a Cortex-A9 and should work on all ARMv7
@ and ARMv6T2 processors.  It has a fast path for short sizes, and has an
@ optimised path for large data sets; the worst case is finding the match early
@ in a large data set.
@ Note: The use of cbz/cbnz means it's Thumb only

@ 2011-07-15 david.gilbert@linaro.org
@    Copy from Cortex strings release 21 and change license
@ http://bazaar.launchpad.net/~linaro-toolchain-dev/cortex-strings/trunk/view/head:/src/linaro-a9/memchr.S
@    Change function declarations/entry/exit
@ 2011-12-01 david.gilbert@linaro.org
@    Add some fixes from comments received (including use of ldrd instead ldm)
@ 2011-12-07 david.gilbert@linaro.org
@    Removed cbz from align loop - can't be taken

@ this lets us check a flag in a 00/ff byte easily in either endianness
#ifdef __ARMEB__
#define CHARTSTMASK(c) 1<<(31-(c*8))
#else
#define CHARTSTMASK(c) 1<<(c*8)
#endif
	.syntax unified

	.text
	.thumb
	.thumb_func
	.global memchr
	.type memchr,%function
ENTRY(memchr)
	@ r0 = start of memory to scan
	@ r1 = character to look for
	@ r2 = length
	@ returns r0 = pointer to character or NULL if not found
	and	r1,r1,#0xff	@ Don't think we can trust the caller to actually pass a char

	cmp	r2,#16		@ If it's short don't bother with anything clever
	blt	20f

	tst	r0, #7		@ If it's already aligned skip the next bit
	beq	10f

	@ Work up to an aligned point
5:
	ldrb	r3, [r0],#1
	subs	r2, r2, #1
	cmp	r3, r1
	beq	50f		@ If it matches exit found
	tst	r0, #7
	bne	5b		@ If not aligned yet then do next byte

10:
	@ At this point, we are aligned, we know we have at least 8 bytes to work with
	push	{r4,r5,r6,r7}
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)

	cfi_remember_state

	orr	r1, r1, r1, lsl #8	@ expand the match word across to all bytes
	orr	r1, r1, r1, lsl #16
	bic	r6, r2, #7	@ Number of double words to work with * 8
	mvns	r7, #0		@ all F's
	movs	r3, #0

15:
	ldrd 	r4,r5, [r0],#8
	subs	r6, r6, #8
	eor	r4,r4, r1	@ Get it so that r4,r5 have 00's where the bytes match the target
	eor	r5,r5, r1
	uadd8	r4, r4, r7	@ Parallel add 0xff - sets the GE bits for anything that wasn't 0
	sel	r4, r3, r7	@ bytes are 00 for none-00 bytes, or ff for 00 bytes - NOTE INVERSION
	uadd8	r5, r5, r7	@ Parallel add 0xff - sets the GE bits for anything that wasn't 0
	sel	r5, r4, r7	@ chained....bytes are 00 for none-00 bytes, or ff for 00 bytes - NOTE INVERSION
	cbnz	r5, 60f
	bne	15b		@ (Flags from the subs above) If not run out of bytes then go around again

	pop	{r4,r5,r6,r7}
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)

	and	r1,r1,#0xff	@ Get r1 back to a single character from the expansion above
	and	r2,r2,#7	@ Leave the count remaining as the number after the double words have been done

20:
	cbz	r2, 40f		@ 0 length or hit the end already then not found

21:  @ Post aligned section, or just a short call
	ldrb	r3,[r0],#1
	subs	r2,r2,#1
	eor	r3,r3,r1	@ r3 = 0 if match - doesn't break flags from sub
	cbz	r3, 50f
	bne	21b		@ on r2 flags

40:
	movs	r0,#0		@ not found
	DO_RET(lr)

50:
	subs	r0,r0,#1	@ found
	DO_RET(lr)

60:  @ We're here because the fast path found a hit - now we have to track down exactly which word it was
     @ r0 points to the start of the double word after the one that was tested
     @ r4 has the 00/ff pattern for the first word, r5 has the chained value
	cfi_restore_state
	cmp	r4, #0
	itte	eq
	moveq	r4, r5		@ the end is in the 2nd word
	subeq	r0,r0,#3	@ Points to 2nd byte of 2nd word
	subne	r0,r0,#7	@ or 2nd byte of 1st word

	@ r0 currently points to the 2nd byte of the word containing the hit
	tst	r4, # CHARTSTMASK(0)	@ 1st character
	bne	61f
	adds	r0,r0,#1
	tst	r4, # CHARTSTMASK(1)	@ 2nd character
	ittt	eq
	addeq	r0,r0,#1
	tsteq	r4, # (3<<15)		@ 2nd & 3rd character
	@ If not the 3rd must be the last one
	addeq	r0,r0,#1

61:
	pop	{r4,r5,r6,r7}
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)

	subs	r0,r0,#1
	DO_RET(lr)

END(memchr)
libc_hidden_builtin_def (memchr)
