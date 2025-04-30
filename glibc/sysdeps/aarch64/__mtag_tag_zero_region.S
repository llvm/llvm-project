/* Copyright (C) 2020-2021 Free Software Foundation, Inc.

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
   <http://www.gnu.org/licenses/>.  */

#include <sysdep.h>

#ifdef USE_MTAG

/* Assumptions:
 *
 * ARMv8-a, AArch64, MTE, LP64 ABI.
 *
 * Interface contract:
 * Address is 16 byte aligned and size is multiple of 16.
 * Returns the passed pointer.
 * The memory region may remain untagged if tagging is not enabled.
 */
	.arch armv8.5-a
	.arch_extension memtag

#define dstin	x0
#define count	x1
#define dst	x2
#define dstend	x3
#define tmp	x4
#define zva_val	x4

ENTRY (__libc_mtag_tag_zero_region)
	PTR_ARG (0)
	SIZE_ARG (1)

	add	dstend, dstin, count

	cmp	count, 96
	b.hi	L(set_long)

	tbnz	count, 6, L(set96)

	/* Set 0, 16, 32, or 48 bytes.  */
	lsr	tmp, count, 5
	add	tmp, dstin, tmp, lsl 4
	cbz     count, L(end)
	stzg	dstin, [dstin]
	stzg	dstin, [tmp]
	stzg	dstin, [dstend, -16]
L(end):
	ret

	.p2align 4
	/* Set 64..96 bytes.  Write 64 bytes from the start and
	   32 bytes from the end.  */
L(set96):
	stz2g	dstin, [dstin]
	stz2g	dstin, [dstin, 32]
	stz2g	dstin, [dstend, -32]
	ret

	.p2align 4
	/* Size is > 96 bytes.  */
L(set_long):
	cmp	count, 160
	b.lo	L(no_zva)

#ifndef SKIP_ZVA_CHECK
	mrs	zva_val, dczid_el0
	and	zva_val, zva_val, 31
	cmp	zva_val, 4		/* ZVA size is 64 bytes.  */
	b.ne	L(no_zva)
#endif
	stz2g	dstin, [dstin]
	stz2g	dstin, [dstin, 32]
	bic	dst, dstin, 63
	sub	count, dstend, dst	/* Count is now 64 too large.  */
	sub	count, count, 128	/* Adjust count and bias for loop.  */

	.p2align 4
L(zva_loop):
	add	dst, dst, 64
	dc	gzva, dst
	subs	count, count, 64
	b.hi	L(zva_loop)
	stz2g	dstin, [dstend, -64]
	stz2g	dstin, [dstend, -32]
	ret

L(no_zva):
	sub	dst, dstin, 32		/* Dst is biased by -32.  */
	sub	count, count, 64	/* Adjust count for loop.  */
L(no_zva_loop):
	stz2g	dstin, [dst, 32]
	stz2g	dstin, [dst, 64]!
	subs	count, count, 64
	b.hi	L(no_zva_loop)
	stz2g	dstin, [dstend, -64]
	stz2g	dstin, [dstend, -32]
	ret

END (__libc_mtag_tag_zero_region)
#endif /* USE_MTAG */
