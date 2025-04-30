/* memchr - find a character in a memory zone using base integer registers

   Copyright (C) 2018-2021 Free Software Foundation, Inc.

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

/* Assumptions:
 *
 * ARMv8-a, AArch64
 * Use base integer registers.
 */

#ifndef MEMCHR
# define MEMCHR __memchr_nosimd
#endif

/* Arguments and results.  */
#define srcin		x0
#define chrin		x1
#define cntin		x2

#define result		x0

#define repchr		x1

#define tmp1		x2
#define tmp2		x3
#define tmp3		x4
#define tmp4		x5

#define src		x6
#define srcend		x7
#define srcend16	x8

#define anymore		x9

#define zeroones	x10

#define data1		x11
#define data2		x12

#define has_chr1	x13
#define has_chr2	x14

#define REP8_01		0x0101010101010101
#define REP8_7f		0x7f7f7f7f7f7f7f7f


ENTRY_ALIGN (MEMCHR, 6)

	PTR_ARG (0)
	SIZE_ARG (2)

	/* Do not dereference srcin if no bytes to compare. */
	cbz	cntin, L(none_chr)

	/* Start address is 16-byte aligned or not? */
	tst	srcin, 15
	bic	src, srcin, 15

	mov	zeroones, REP8_01
	and	repchr, chrin, 255
	/* Generate a qword integer as |c|c|c|c|c|c|c|c|. */
	mul	repchr, repchr, zeroones

	add	srcend, srcin, cntin
	/*
	 * srcend16 is address of the block following the last block.
	 *
	 * [A block is 16-byte aligned and sized.]
	 */
	add	srcend16, srcend, 15
	bic	srcend16, srcend16, 15

	b.eq	L(loop)

	/* Load the first block containing start address. */
	ldp	data1, data2, [src], 16

	lsl	tmp1, srcin, 3
	mov	tmp2, ~0
#ifdef __AARCH64EB__
	lsr	tmp3, tmp2, tmp1
#else
	lsl	tmp3, tmp2, tmp1
#endif
	/* Start address is in the first or the second qword? */
	tst	srcin, 8

	/*
	 * Transform any byte in the block to zero using XOR operation,
	 * if that byte equals the char to search. In this way, searching
	 * the char becomes detecting zero in the resulting two qwords.
	 */
	eor	data1, data1, repchr
	eor	data2, data2, repchr

	/*
	 * Set those unused bytes(before start address) to 0xff, so
	 * that they will not hit any zero detection.
	 */
	orn	tmp1, data1, tmp3
	orn	tmp2, data2, tmp3

	csinv	data1, tmp1, xzr, eq
	csel	data2, data2, tmp2, eq

	/*
	 * When the first and last block are the same, there are two cases:
	 *  o. Memory range to search is just in one block.
	 *      ( start address - end address) < 0
	 *
	 *  o. Memory range is so large that end address wrap-around.
	 *      ( start address - end address) > 0
	 */
	cmp	srcin, srcend
	ccmp	src, srcend16, 0, mi
	csetm	anymore, ne
	b	L(find_chr)

	.p2align 4
L(loop):
	ldp	data1, data2, [src], 16

	subs	anymore, src, srcend16

	/*
	 * Transform any byte in the block to zero using XOR operation,
	 * if that byte equals the char to search.
	 */
	eor	data1, data1, repchr
	eor	data2, data2, repchr

L(find_chr):
	/*
	 * Use the following integer test to find out if any byte in a
	 * qword is zero. If do not contain zero-valued byte, test result
	 * is zero.
	 *
	 *  (qword - 0x0101010101010101) & ~(qword) & 0x8080808080808080
	 * =
	 *  (qword - 0x0101010101010101) & ~(qword  | 0x7f7f7f7f7f7f7f7f)
	 *
	 */
	sub	tmp1, data1, zeroones
	sub	tmp2, data2, zeroones

	orr	tmp3, data1, REP8_7f
	orr	tmp4, data2, REP8_7f

	bic	has_chr1, tmp1, tmp3
	bic	has_chr2, tmp2, tmp4

	orr	tmp1, has_chr1, has_chr2
	ccmp	tmp1, 0, 0, ne

	b.eq	L(loop)

	cbz	has_chr1, 1f
	sub	result, src, 16
#ifdef __AARCH64EB__
	rev	data1, data1
#else
	rev	has_chr1, has_chr1
#endif
	b	L(done)

1:	cbz	has_chr2, L(none_chr)
	sub	result, src, 8
#ifdef __AARCH64EB__
	rev	data1, data2
#else
	rev	has_chr1, has_chr2
#endif

L(done):
#ifdef __AARCH64EB__
	/*
	 * For big-endian, can not directly use has_chr1/has_chr2 because
	 * two qwords has been reversed after loading from memory.
	 * Thus, have to perform char detection on two qwords again, which
	 * should be byte-swapped this time.
	 */
	sub	tmp1, data1, zeroones
	orr	tmp3, data1, REP8_7f
	bic	has_chr1, tmp1, tmp3
	rev	has_chr1, has_chr1
#endif

	/*
	 * If the specified char is found in a qword, the corresponding
	 * byte of in has_chr has value of 1, while this is only true for
	 * the first occurrence, not other occurrences.
	 */
	cmp	anymore, 0
	clz	tmp1, has_chr1
	add	result, result, tmp1, lsr 3
	ccmp	result, srcend, 8, eq	/* NZCV = 8000 */
	csel	result, result, xzr, mi
	ret

L(none_chr):
	mov	result, 0
	ret

END (MEMCHR)
libc_hidden_builtin_def (MEMCHR)
