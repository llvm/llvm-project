/* memcmp - compare memory

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

/* Assumptions:
 *
 * ARMv8-a, AArch64, unaligned accesses.
 */

/* Parameters and result.  */
#define src1		x0
#define src2		x1
#define limit		x2
#define result		w0

/* Internal variables.  */
#define data1		x3
#define data1w		w3
#define data1h		x4
#define data2		x5
#define data2w		w5
#define data2h		x6
#define tmp1		x7
#define tmp2		x8

ENTRY_ALIGN (memcmp, 6)
	PTR_ARG (0)
	PTR_ARG (1)
	SIZE_ARG (2)

	subs	limit, limit, 16
	b.lo	L(less16)

	ldp	data1, data1h, [src1], 16
	ldp	data2, data2h, [src2], 16
	ccmp	data1, data2, 0, ne
	ccmp	data1h, data2h, 0, eq
	b.ne	L(return64)

	subs	limit, limit, 16
	b.ls	L(last_bytes)
	cmp	limit, 112
	b.lo	L(loop16)

	and	tmp1, src1, 15
	add	limit, limit, tmp1
	sub	src1, src1, tmp1
	sub	src2, src2, tmp1
	subs	limit, limit, 48

	/* Compare 128 up bytes using aligned access. */
	.p2align 4
L(loop64):
	ldp	data1, data1h, [src1]
	ldp	data2, data2h, [src2]
	cmp	data1, data2
	ccmp	data1h, data2h, 0, eq
	b.ne	L(return64)

	ldp	data1, data1h, [src1, 16]
	ldp	data2, data2h, [src2, 16]
	cmp	data1, data2
	ccmp	data1h, data2h, 0, eq
	b.ne	L(return64)

	ldp	data1, data1h, [src1, 32]
	ldp	data2, data2h, [src2, 32]
	cmp	data1, data2
	ccmp	data1h, data2h, 0, eq
	b.ne	L(return64)

	ldp	data1, data1h, [src1, 48]
	ldp	data2, data2h, [src2, 48]
	cmp	data1, data2
	ccmp	data1h, data2h, 0, eq
	b.ne	L(return64)

	subs	limit, limit, 64
	add	src1, src1, 64
	add	src2, src2, 64
	b.pl	L(loop64)
	adds	limit, limit, 48
	b.lo	L(last_bytes)

L(loop16):
	ldp	data1, data1h, [src1], 16
	ldp	data2, data2h, [src2], 16
	cmp	data1, data2
	ccmp	data1h, data2h, 0, eq
	b.ne	L(return64)

	subs	limit, limit, 16
	b.hi	L(loop16)
	/* Compare last 1-16 bytes using unaligned access.  */
L(last_bytes):
	add	src1, src1, limit
	add	src2, src2, limit
	ldp	data1, data1h, [src1]
	ldp	data2, data2h, [src2]

	/* Compare data bytes and set return value to 0, -1 or 1.  */
L(return64):
	cmp	data1, data2
	csel	data1, data1, data1h, ne
	csel	data2, data2, data2h, ne
L(return):
#ifndef __AARCH64EB__
	rev	data1, data1
	rev	data2, data2
#endif
	cmp	data1, data2
	cset	result, ne
	cneg	result, result, lo
	ret

	.p2align 4
L(less16):
	adds	limit, limit, 8
	b.lo	L(less8)		//lo:<
	ldr	data1, [src1]
	ldr	data2, [src2]
	/* equal 8 optimized */
	ccmp	data1, data2, 0, ne
	b.ne	L(return)

	ldr	data1, [src1, limit]
	ldr	data2, [src2, limit]
	b	L(return)

	.p2align 4
L(less8):
	adds	limit, limit, 4
	b.lo	L(less4)
	ldr	data1w, [src1]
	ldr	data2w, [src2]
	ccmp	data1w, data2w, 0, ne
	b.ne	L(return)
	ldr	data1w,	[src1, limit]
	ldr	data2w,	[src2, limit]
	b	L(return)

	.p2align 4
L(less4):
	adds	limit, limit, 4
	b.eq	L(ret_0)

L(byte_loop):
	ldrb	data1w, [src1], 1
	ldrb	data2w, [src2], 1
	subs	limit, limit, 1
	ccmp	data1w, data2w, 0, ne	/* NZCV = 0b0000.  */
	b.eq	L(byte_loop)
	sub	result, data1w, data2w
	ret
L(ret_0):
	mov	result, 0
	ret

END (memcmp)
#undef bcmp
weak_alias (memcmp, bcmp)
libc_hidden_builtin_def (memcmp)
