/* strchr - find a character in a string

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
 * ARMv8-a, AArch64, Advanced SIMD.
 * MTE compatible.
 */

#define srcin		x0
#define chrin		w1
#define result		x0

#define src		x2
#define tmp1		x1
#define wtmp2		w3
#define tmp3		x3

#define vrepchr		v0
#define vdata		v1
#define qdata		q1
#define vhas_nul	v2
#define vhas_chr	v3
#define vrepmask	v4
#define vrepmask2	v5
#define vend		v6
#define dend		d6

/* Core algorithm.

   For each 16-byte chunk we calculate a 64-bit syndrome value with four bits
   per byte. For even bytes, bits 0-1 are set if the relevant byte matched the
   requested character, bits 2-3 are set if the byte is NUL (or matched), and
   bits 4-7 are not used and must be zero if none of bits 0-3 are set). Odd
   bytes set bits 4-7 so that adjacent bytes can be merged. Since the bits
   in the syndrome reflect the order in which things occur in the original
   string, counting trailing zeros identifies exactly which byte matched.  */

ENTRY (strchr)
	PTR_ARG (0)
	bic	src, srcin, 15
	dup	vrepchr.16b, chrin
	ld1	{vdata.16b}, [src]
	mov	wtmp2, 0x3003
	dup	vrepmask.8h, wtmp2
	cmeq	vhas_nul.16b, vdata.16b, 0
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	mov	wtmp2, 0xf00f
	dup	vrepmask2.8h, wtmp2

	bit	vhas_nul.16b, vhas_chr.16b, vrepmask.16b
	and	vhas_nul.16b, vhas_nul.16b, vrepmask2.16b
	lsl	tmp3, srcin, 2
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b		/* 128->64 */

	fmov	tmp1, dend
	lsr	tmp1, tmp1, tmp3
	cbz	tmp1, L(loop)

	rbit	tmp1, tmp1
	clz	tmp1, tmp1
	/* Tmp1 is an even multiple of 2 if the target character was
	   found first. Otherwise we've found the end of string.  */
	tst	tmp1, 2
	add	result, srcin, tmp1, lsr 2
	csel	result, result, xzr, eq
	ret

	.p2align 4
L(loop):
	ldr	qdata, [src, 16]!
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	cmhs	vhas_nul.16b, vhas_chr.16b, vdata.16b
	umaxp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	tmp1, dend
	cbz	tmp1, L(loop)

#ifdef __AARCH64EB__
	bif	vhas_nul.16b, vhas_chr.16b, vrepmask.16b
	and	vhas_nul.16b, vhas_nul.16b, vrepmask2.16b
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b		/* 128->64 */
	fmov	tmp1, dend
#else
	bit	vhas_nul.16b, vhas_chr.16b, vrepmask.16b
	and	vhas_nul.16b, vhas_nul.16b, vrepmask2.16b
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b		/* 128->64 */
	fmov	tmp1, dend
	rbit	tmp1, tmp1
#endif
	clz	tmp1, tmp1
	/* Tmp1 is an even multiple of 2 if the target character was
	   found first. Otherwise we've found the end of string.  */
	tst	tmp1, 2
	add	result, src, tmp1, lsr 2
	csel	result, result, xzr, eq
	ret

END (strchr)
libc_hidden_builtin_def (strchr)
weak_alias (strchr, index)
