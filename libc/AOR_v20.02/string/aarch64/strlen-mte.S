/*
 * strlen - calculate the length of a string
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* Assumptions:
 *
 * ARMv8-a, AArch64.
 */

#include "../asmdefs.h"

/* Arguments and results.  */
#define srcin		x0
#define len		x0

/* Locals and temporaries.  */
#define src		x1
#define data1		x2
#define data2		x3
#define has_nul1	x4
#define has_nul2	x5
#define tmp1		x4
#define tmp2		x5
#define tmp3		x6
#define tmp4		x7
#define zeroones	x8
#define offset		x9

	/* NUL detection works on the principle that (X - 1) & (~X) & 0x80
	   (=> (X - 1) & ~(X | 0x7f)) is non-zero iff a byte is zero, and
	   can be done in parallel across the entire word. A faster check
	   (X - 1) & 0x80 is zero for non-NUL ASCII characters, but gives
	   false hits for characters 129..255.	*/

#define REP8_01 0x0101010101010101
#define REP8_7f 0x7f7f7f7f7f7f7f7f

	/* This implementation is compatible with Memory Tagging. All loads
	   are 16 bytes in size and 16 bytes aligned. This also avoids the
	   need for page boundary checks. This implementation is correct
	   even without Memory Tagging, but other implementations could be
	   more beneficial if Memory Tagging is not enabled.

	   First load is aligned down and can contain bytes that are located
	   before the string. This is handled by modifying the "zeroones"
	   mask. The bytes that need to be ignored are set to zero.
	   If the string is aligned in such a way that 8 or more bytes from
	   the first load should be ignored, there is a special case
	   (skip_first_8_bytes) which only compares the second 8 bytes.

	   If there is a NUL byte in the first load, we calculate the length
	   from the 2 8-byte words using conditional select to reduce branch
	   mispredictions.

	   If the string is longer than 16 bytes, we check 32 bytes per
	   iteration using the fast NUL check (main_loop). If we encounter
	   non-ASCII characters, we fallback to a second loop
	   (nonascii_loop) using the full NUL check.  */

ENTRY(__strlen_aarch64_mte)
	bic	src, srcin, 15	/* Align down to 16 bytes.  */
	mov	zeroones, REP8_01
	/* (offset & 63) holds number of bits to ignore in a register.*/
	lsl	offset, srcin, 3
	ldp	data1, data2, [src], -16
	lsl	tmp1, zeroones, offset	/* Shift (offset & 63).  */
#ifdef __AARCH64EB__
	/* For big-endian, carry propagation (if the final byte in the
	   string is 0x01) means we cannot use has_nul1/2 directly.
	   e.g. 0x0100 - 0x0101 = 0xffff, so 0x01 will be mistaken for NUL.
	   Since we expect strings to be small and early-exit,
	   byte-swap the data now so has_null1/2 will be correct.  */
	rev	data1, data1
	rev	data2, data2
#endif
	tbnz	srcin, 3, L(skip_first_8_bytes)
	sub	tmp1, data1, tmp1
	orr	tmp2, data1, REP8_7f
	sub	tmp3, data2, zeroones
	orr	tmp4, data2, REP8_7f
	bics	has_nul1, tmp1, tmp2
	bic	has_nul2, tmp3, tmp4
	/* If comparison happens, C flag is always set. */
	ccmp	has_nul2, 0, 0, eq
	beq	L(main_loop)

	/* Enter with C = has_nul1 == 0.  */
	csel	has_nul1, has_nul1, has_nul2, cc
	and	tmp2, srcin, 7	/* Bytes to ignore. */
	rev	has_nul1, has_nul1
	neg	tmp2, tmp2
	clz	tmp1, has_nul1	/* Count bits before NUL. */
	/* Add 8 if NUL byte is not in first register. */
	add	tmp3, tmp2, 8
	csel	len, tmp2, tmp3, cc
	add	len, len, tmp1, lsr 3
	ret

L(skip_first_8_bytes):
	sub	tmp1, data2, tmp1
	orr	tmp2, data2, REP8_7f
	bics	has_nul1, tmp1, tmp2
	beq	L(main_loop)

	rev	has_nul1, has_nul1
	lsl	tmp1, has_nul1, offset	/* Ignore bytes before string. */
	clz	tmp1, tmp1	/* Count bits before NUL. */
	lsr	len, tmp1, 3
	ret

	/* The inner loop processes 32 bytes per iteration and uses the fast
	   NUL check.  If we encounter non-ASCII characters, use a second
	   loop with the accurate NUL check.  */
	.p2align 4
L(main_loop):
	ldp	data1, data2, [src, 32]!
	sub	tmp1, data1, zeroones
	sub	tmp3, data2, zeroones
	orr	tmp2, tmp1, tmp3
	tst	tmp2, zeroones, lsl 7
	bne	1f
	ldp	data1, data2, [src, 16]
	sub	tmp1, data1, zeroones
	sub	tmp3, data2, zeroones
	orr	tmp2, tmp1, tmp3
	tst	tmp2, zeroones, lsl 7
	beq	L(main_loop)
	add	src, src, 16
1:
	/* The fast check failed, so do the slower, accurate NUL check.	 */
	orr	tmp2, data1, REP8_7f
	orr	tmp4, data2, REP8_7f
	bics	has_nul1, tmp1, tmp2
	bic	has_nul2, tmp3, tmp4
	ccmp	has_nul2, 0, 0, eq
	beq	L(nonascii_loop)

	/* Enter with C = has_nul1 == 0.  */
L(tail):
#ifdef __AARCH64EB__
	/* For big-endian, carry propagation (if the final byte in the
	   string is 0x01) means we cannot use has_nul1/2 directly.  The
	   easiest way to get the correct byte is to byte-swap the data
	   and calculate the syndrome a second time.  */
	csel	data1, data1, data2, cc
	rev	data1, data1
	sub	tmp1, data1, zeroones
	orr	tmp2, data1, REP8_7f
	bic	has_nul1, tmp1, tmp2
#else
	csel	has_nul1, has_nul1, has_nul2, cc
#endif
	sub	len, src, srcin
	rev	has_nul1, has_nul1
	add	tmp2, len, 8
	clz	tmp1, has_nul1
	csel	len, len, tmp2, cc
	add	len, len, tmp1, lsr 3
	ret

L(nonascii_loop):
	ldp	data1, data2, [src, 16]!
	sub	tmp1, data1, zeroones
	orr	tmp2, data1, REP8_7f
	sub	tmp3, data2, zeroones
	orr	tmp4, data2, REP8_7f
	bics	has_nul1, tmp1, tmp2
	bic	has_nul2, tmp3, tmp4
	ccmp	has_nul2, 0, 0, eq
	bne	L(tail)
	ldp	data1, data2, [src, 16]!
	sub	tmp1, data1, zeroones
	orr	tmp2, data1, REP8_7f
	sub	tmp3, data2, zeroones
	orr	tmp4, data2, REP8_7f
	bics	has_nul1, tmp1, tmp2
	bic	has_nul2, tmp3, tmp4
	ccmp	has_nul2, 0, 0, eq
	beq	L(nonascii_loop)
	b	L(tail)

END(__strlen_aarch64_mte)
