/* Optimized memset for Huawei Kunpeng processor.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.

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
#include <sysdeps/aarch64/memset-reg.h>

#if IS_IN (libc)
# define MEMSET __memset_kunpeng

/* Assumptions:
 *
 * ARMv8-a, AArch64, unaligned accesses
 *
 */

ENTRY_ALIGN (MEMSET, 6)

	PTR_ARG (0)
	SIZE_ARG (2)

	dup	v0.16B, valw
	add	dstend, dstin, count

	cmp	count, 128
	b.hs	L(set_long)

	cmp	count, 16
	b.lo	L(less16)

	/* Set 16..127 bytes.  */
	str	q0, [dstin]
	tbnz	count, 6, L(set127)
	str	q0, [dstend, -16]
	tbz	count, 5, 1f
	str	q0, [dstin, 16]
	str	q0, [dstend, -32]
1:	ret

	.p2align 4
	/* Set 64..127 bytes.  Write 64 bytes from the start and
	   64 bytes from the end.  */
L(set127):
	stp	q0, q0, [dstin, 16]
	str	q0, [dstin, 48]
	stp	q0, q0, [dstend, -64]
	stp	q0, q0, [dstend, -32]
	ret

	.p2align 4
	/* Set 0..15 bytes.  */
L(less16):
	tbz	count, 3, L(less8)
	str	d0, [dstin]
	str	d0, [dstend, -8]
	ret
L(less8):
	tbz	count, 2, 2f
	str	s0, [dstin]
	str	s0, [dstend, -4]
	ret
2:	cbz	count, 3f
	str	b0, [dstin]
	tbz	count, 1, 3f
	str	h0, [dstend, -2]
3:	ret

	.p2align 4
L(set_long):
	bic	dst, dstin, 15
	str	q0, [dstin]
	sub	count, dstend, dst	/* Count is 16 too large.  */
	sub	dst, dst, 16		/* Dst is biased by -32.  */
	sub	count, count, 64 + 16 + 1 /* Adjust count and bias for loop.  */
1:	stp	q0, q0, [dst, 32]
	stp	q0, q0, [dst, 64]!
	subs	count, count, 64
	b.lo	1f
	stp	q0, q0, [dst, 32]
	stp	q0, q0, [dst, 64]!
	subs	count, count, 64
	b.lo	1f
	stp	q0, q0, [dst, 32]
	stp	q0, q0, [dst, 64]!
	subs	count, count, 64
	b.lo	1f
	stp	q0, q0, [dst, 32]
	stp	q0, q0, [dst, 64]!
	subs	count, count, 64
	b.hs	1b

1:	stp	q0, q0, [dstend, -64]
	stp	q0, q0, [dstend, -32]
	ret

END (MEMSET)
libc_hidden_builtin_def (MEMSET)
#endif
