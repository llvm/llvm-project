/* strcpy/stpcpy - copy a string returning pointer to start/end.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* To build as stpcpy, define BUILD_STPCPY before compiling this file.

   To test the page crossing code path more thoroughly, compile with
   -DSTRCPY_TEST_PAGE_CROSS - this will force all unaligned copies through
   the slower entry path.  This option is not intended for production use.  */

#include <sysdep.h>

/* Assumptions:
 *
 * ARMv8-a, AArch64, Advanced SIMD.
 * MTE compatible.
 */

/* Arguments and results.  */
#define dstin		x0
#define srcin		x1
#define result		x0

#define src		x2
#define dst		x3
#define len		x4
#define synd		x4
#define	tmp		x5
#define wtmp		w5
#define shift		x5
#define data1		x6
#define dataw1		w6
#define data2		x7
#define dataw2		w7

#define dataq		q0
#define vdata		v0
#define vhas_nul	v1
#define vrepmask	v2
#define vend		v3
#define dend		d3
#define dataq2		q1

#ifdef BUILD_STPCPY
# define STRCPY __stpcpy
# define IFSTPCPY(X,...) X,__VA_ARGS__
#else
# define STRCPY strcpy
# define IFSTPCPY(X,...)
#endif

/* Core algorithm:

   For each 16-byte chunk we calculate a 64-bit syndrome value with four bits
   per byte. For even bytes, bits 0-3 are set if the relevant byte matched the
   requested character or the byte is NUL. Bits 4-7 must be zero. Bits 4-7 are
   set likewise for odd bytes so that adjacent bytes can be merged. Since the
   bits in the syndrome reflect the order in which things occur in the original
   string, counting trailing zeros identifies exactly which byte matched.  */

ENTRY (STRCPY)
	PTR_ARG (0)
	PTR_ARG (1)
	bic	src, srcin, 15
	mov	wtmp, 0xf00f
	ld1	{vdata.16b}, [src]
	dup	vrepmask.8h, wtmp
	cmeq	vhas_nul.16b, vdata.16b, 0
	lsl	shift, srcin, 2
	and	vhas_nul.16b, vhas_nul.16b, vrepmask.16b
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	lsr	synd, synd, shift
	cbnz	synd, L(tail)

	ldr	dataq, [src, 16]!
	cmeq	vhas_nul.16b, vdata.16b, 0
	and	vhas_nul.16b, vhas_nul.16b, vrepmask.16b
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	cbz	synd, L(start_loop)

#ifndef __AARCH64EB__
	rbit	synd, synd
#endif
	sub	tmp, src, srcin
	clz	len, synd
	add	len, tmp, len, lsr 2
	tbz	len, 4, L(less16)
	sub	tmp, len, 15
	ldr	dataq, [srcin]
	ldr	dataq2, [srcin, tmp]
	str	dataq, [dstin]
	str	dataq2, [dstin, tmp]
	IFSTPCPY (add result, dstin, len)
	ret

	.p2align 4,,8
L(tail):
	rbit	synd, synd
	clz	len, synd
	lsr	len, len, 2

	.p2align 4
L(less16):
	tbz	len, 3, L(less8)
	sub	tmp, len, 7
	ldr	data1, [srcin]
	ldr	data2, [srcin, tmp]
	str	data1, [dstin]
	str	data2, [dstin, tmp]
	IFSTPCPY (add result, dstin, len)
	ret

	.p2align 4
L(less8):
	subs	tmp, len, 3
	b.lo	L(less4)
	ldr	dataw1, [srcin]
	ldr	dataw2, [srcin, tmp]
	str	dataw1, [dstin]
	str	dataw2, [dstin, tmp]
	IFSTPCPY (add result, dstin, len)
	ret

L(less4):
	cbz	len, L(zerobyte)
	ldrh	dataw1, [srcin]
	strh	dataw1, [dstin]
L(zerobyte):
	strb	wzr, [dstin, len]
	IFSTPCPY (add result, dstin, len)
	ret

	.p2align 4
L(start_loop):
	sub	len, src, srcin
	ldr	dataq2, [srcin]
	add	dst, dstin, len
	str	dataq2, [dstin]

	.p2align 5
L(loop):
	str	dataq, [dst], 16
	ldr	dataq, [src, 16]!
	cmeq	vhas_nul.16b, vdata.16b, 0
	umaxp	vend.16b, vhas_nul.16b, vhas_nul.16b
	fmov	synd, dend
	cbz	synd, L(loop)

	and	vhas_nul.16b, vhas_nul.16b, vrepmask.16b
	addp	vend.16b, vhas_nul.16b, vhas_nul.16b		/* 128->64 */
	fmov	synd, dend
#ifndef __AARCH64EB__
	rbit	synd, synd
#endif
	clz	len, synd
	lsr	len, len, 2
	sub	tmp, len, 15
	ldr	dataq, [src, tmp]
	str	dataq, [dst, tmp]
	IFSTPCPY (add result, dst, len)
	ret

END (STRCPY)

#ifdef BUILD_STPCPY
weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
#else
libc_hidden_builtin_def (strcpy)
#endif
