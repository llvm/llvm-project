/* strrchr: find the last instance of a character in a string.

   Copyright (C) 2014-2021 Free Software Foundation, Inc.

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
 * Neon Available.
 * MTE compatible.
 */

/* Arguments and results.  */
#define srcin		x0
#define chrin		w1
#define result		x0

#define src		x2
#define tmp		x3
#define wtmp		w3
#define synd		x3
#define shift		x4
#define src_match	x4
#define nul_match	x5
#define chr_match	x6

#define vrepchr		v0
#define vdata		v1
#define vhas_nul	v2
#define vhas_chr	v3
#define vrepmask	v4
#define vrepmask2	v5
#define vend		v5
#define dend		d5

/* Core algorithm.

   For each 16-byte chunk we calculate a 64-bit syndrome value, with
   four bits per byte (LSB is always in bits 0 and 1, for both big
   and little-endian systems).  For each tuple, bits 0-1 are set if
   the relevant byte matched the requested character; bits 2-3 are set
   if the relevant byte matched the NUL end of string.  */

ENTRY(strrchr)
	PTR_ARG (0)
	bic	src, srcin, 15
	dup	vrepchr.16b, chrin
	mov	wtmp, 0x3003
	dup	vrepmask.8h, wtmp
	tst	srcin, 15
	beq	L(loop1)

	ld1	{vdata.16b}, [src], 16
	cmeq	vhas_nul.16b, vdata.16b, 0
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	mov	wtmp, 0xf00f
	dup	vrepmask2.8h, wtmp
	bit	vhas_nul.16b, vhas_chr.16b, vrepmask.16b
	and	vhas_nul.16b, vhas_nul.16b, vrepmask2.16b
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b
	lsl	shift, srcin, 2
	fmov	synd, dend
	lsr	synd, synd, shift
	lsl	synd, synd, shift
	ands	nul_match, synd, 0xcccccccccccccccc
	bne	L(tail)
	cbnz	synd, L(loop2)

	.p2align 5
L(loop1):
	ld1	{vdata.16b}, [src], 16
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	cmhs	vhas_nul.16b, vhas_chr.16b, vdata.16b
	umaxp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	cbz	synd, L(loop1)

	cmeq	vhas_nul.16b, vdata.16b, 0
	bit	vhas_nul.16b, vhas_chr.16b, vrepmask.16b
	bic	vhas_nul.8h, 0x0f, lsl 8
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	ands	nul_match, synd, 0xcccccccccccccccc
	beq	L(loop2)

L(tail):
	sub	nul_match, nul_match, 1
	and	chr_match, synd, 0x3333333333333333
	ands	chr_match, chr_match, nul_match
	sub	result, src, 1
	clz	tmp, chr_match
	sub	result, result, tmp, lsr 2
	csel	result, result, xzr, ne
	ret

	.p2align 4
L(loop2):
	cmp	synd, 0
	csel	src_match, src, src_match, ne
	csel	chr_match, synd, chr_match, ne
	ld1	{vdata.16b}, [src], 16
	cmeq	vhas_nul.16b, vdata.16b, 0
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	bit	vhas_nul.16b, vhas_chr.16b, vrepmask.16b
	umaxp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	tst	synd, 0xcccccccccccccccc
	beq	L(loop2)

	bic	vhas_nul.8h, 0x0f, lsl 8
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	and	nul_match, synd, 0xcccccccccccccccc
	sub	nul_match, nul_match, 1
	and	tmp, synd, 0x3333333333333333
	ands	tmp, tmp, nul_match
	csel	chr_match, tmp, chr_match, ne
	csel	src_match, src, src_match, ne
	sub	src_match, src_match, 1
	clz	tmp, chr_match
	sub	result, src_match, tmp, lsr 2
	ret

END(strrchr)
weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
