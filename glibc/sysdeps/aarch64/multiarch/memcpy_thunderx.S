/* A Thunderx Optimized memcpy implementation for AARCH64.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

/* The actual code in this memcpy and memmove should be identical to the
   generic version except for the code under '#ifdef THUNDERX'.  This is
   to make is easier to keep this version and the generic version in sync
   for changes that are not specific to thunderx.  */

#include <sysdep.h>

/* Assumptions:
 *
 * ARMv8-a, AArch64, unaligned accesses.
 *
 */

#define dstin	x0
#define src	x1
#define count	x2
#define dst	x3
#define srcend	x4
#define dstend	x5
#define A_l	x6
#define A_lw	w6
#define A_h	x7
#define A_hw	w7
#define B_l	x8
#define B_lw	w8
#define B_h	x9
#define C_l	x10
#define C_h	x11
#define D_l	x12
#define D_h	x13
#define E_l	src
#define E_h	count
#define F_l	srcend
#define F_h	dst
#define G_l	count
#define G_h	dst
#define tmp1	x14

/* Copies are split into 3 main cases: small copies of up to 16 bytes,
   medium copies of 17..96 bytes which are fully unrolled. Large copies
   of more than 96 bytes align the destination and use an unrolled loop
   processing 64 bytes per iteration.
   In order to share code with memmove, small and medium copies read all
   data before writing, allowing any kind of overlap. So small, medium
   and large backwards memmoves are handled by falling through into memcpy.
   Overlapping large forward memmoves use a loop that copies backwards.
*/

#ifndef MEMMOVE
# define MEMMOVE memmove
#endif
#ifndef MEMCPY
# define MEMCPY memcpy
#endif

#if IS_IN (libc)

#  undef MEMCPY
#  define MEMCPY __memcpy_thunderx
#  undef MEMMOVE
#  define MEMMOVE __memmove_thunderx

ENTRY_ALIGN (MEMMOVE, 6)

	PTR_ARG (0)
	PTR_ARG (1)
	SIZE_ARG (2)

	sub	tmp1, dstin, src
	cmp	count, 96
	ccmp	tmp1, count, 2, hi
	b.lo	L(move_long)

	/* Common case falls through into memcpy.  */
END (MEMMOVE)
libc_hidden_builtin_def (MEMMOVE)
ENTRY (MEMCPY)

	PTR_ARG (0)
	PTR_ARG (1)
	SIZE_ARG (2)

	prfm	PLDL1KEEP, [src]
	add	srcend, src, count
	add	dstend, dstin, count
	cmp	count, 16
	b.ls	L(copy16)
	cmp	count, 96
	b.hi	L(copy_long)

	/* Medium copies: 17..96 bytes.  */
	sub	tmp1, count, 1
	ldp	A_l, A_h, [src]
	tbnz	tmp1, 6, L(copy96)
	ldp	D_l, D_h, [srcend, -16]
	tbz	tmp1, 5, 1f
	ldp	B_l, B_h, [src, 16]
	ldp	C_l, C_h, [srcend, -32]
	stp	B_l, B_h, [dstin, 16]
	stp	C_l, C_h, [dstend, -32]
1:
	stp	A_l, A_h, [dstin]
	stp	D_l, D_h, [dstend, -16]
	ret

	.p2align 4
	/* Small copies: 0..16 bytes.  */
L(copy16):
	cmp	count, 8
	b.lo	1f
	ldr	A_l, [src]
	ldr	A_h, [srcend, -8]
	str	A_l, [dstin]
	str	A_h, [dstend, -8]
	ret
	.p2align 4
1:
	tbz	count, 2, 1f
	ldr	A_lw, [src]
	ldr	A_hw, [srcend, -4]
	str	A_lw, [dstin]
	str	A_hw, [dstend, -4]
	ret

	/* Copy 0..3 bytes.  Use a branchless sequence that copies the same
	   byte 3 times if count==1, or the 2nd byte twice if count==2.  */
1:
	cbz	count, 2f
	lsr	tmp1, count, 1
	ldrb	A_lw, [src]
	ldrb	A_hw, [srcend, -1]
	ldrb	B_lw, [src, tmp1]
	strb	A_lw, [dstin]
	strb	B_lw, [dstin, tmp1]
	strb	A_hw, [dstend, -1]
2:	ret

	.p2align 4
	/* Copy 64..96 bytes.  Copy 64 bytes from the start and
	   32 bytes from the end.  */
L(copy96):
	ldp	B_l, B_h, [src, 16]
	ldp	C_l, C_h, [src, 32]
	ldp	D_l, D_h, [src, 48]
	ldp	E_l, E_h, [srcend, -32]
	ldp	F_l, F_h, [srcend, -16]
	stp	A_l, A_h, [dstin]
	stp	B_l, B_h, [dstin, 16]
	stp	C_l, C_h, [dstin, 32]
	stp	D_l, D_h, [dstin, 48]
	stp	E_l, E_h, [dstend, -32]
	stp	F_l, F_h, [dstend, -16]
	ret

	/* Align DST to 16 byte alignment so that we don't cross cache line
	   boundaries on both loads and stores.  There are at least 96 bytes
	   to copy, so copy 16 bytes unaligned and then align.  The loop
	   copies 64 bytes per iteration and prefetches one iteration ahead.  */

	.p2align 4
L(copy_long):

	/* On thunderx, large memcpy's are helped by software prefetching.
	   This loop is identical to the one below it but with prefetching
	   instructions included.  For loops that are less than 32768 bytes,
	   the prefetching does not help and slow the code down so we only
	   use the prefetching loop for the largest memcpys.  */

	cmp	count, #32768
	b.lo	L(copy_long_without_prefetch)
	and	tmp1, dstin, 15
	bic	dst, dstin, 15
	ldp	D_l, D_h, [src]
	sub	src, src, tmp1
	prfm	pldl1strm, [src, 384]
	add	count, count, tmp1	/* Count is now 16 too large.  */
	ldp	A_l, A_h, [src, 16]
	stp	D_l, D_h, [dstin]
	ldp	B_l, B_h, [src, 32]
	ldp	C_l, C_h, [src, 48]
	ldp	D_l, D_h, [src, 64]!
	subs	count, count, 128 + 16	/* Test and readjust count.  */

L(prefetch_loop64):
	tbz	src, #6, 1f
	prfm	pldl1strm, [src, 512]
1:
	stp	A_l, A_h, [dst, 16]
	ldp	A_l, A_h, [src, 16]
	stp	B_l, B_h, [dst, 32]
	ldp	B_l, B_h, [src, 32]
	stp	C_l, C_h, [dst, 48]
	ldp	C_l, C_h, [src, 48]
	stp	D_l, D_h, [dst, 64]!
	ldp	D_l, D_h, [src, 64]!
	subs	count, count, 64
	b.hi	L(prefetch_loop64)
	b	L(last64)

L(copy_long_without_prefetch):

	and	tmp1, dstin, 15
	bic	dst, dstin, 15
	ldp	D_l, D_h, [src]
	sub	src, src, tmp1
	add	count, count, tmp1	/* Count is now 16 too large.  */
	ldp	A_l, A_h, [src, 16]
	stp	D_l, D_h, [dstin]
	ldp	B_l, B_h, [src, 32]
	ldp	C_l, C_h, [src, 48]
	ldp	D_l, D_h, [src, 64]!
	subs	count, count, 128 + 16	/* Test and readjust count.  */
	b.ls	L(last64)
L(loop64):
	stp	A_l, A_h, [dst, 16]
	ldp	A_l, A_h, [src, 16]
	stp	B_l, B_h, [dst, 32]
	ldp	B_l, B_h, [src, 32]
	stp	C_l, C_h, [dst, 48]
	ldp	C_l, C_h, [src, 48]
	stp	D_l, D_h, [dst, 64]!
	ldp	D_l, D_h, [src, 64]!
	subs	count, count, 64
	b.hi	L(loop64)

	/* Write the last full set of 64 bytes.  The remainder is at most 64
	   bytes, so it is safe to always copy 64 bytes from the end even if
	   there is just 1 byte left.  */
L(last64):
	ldp	E_l, E_h, [srcend, -64]
	stp	A_l, A_h, [dst, 16]
	ldp	A_l, A_h, [srcend, -48]
	stp	B_l, B_h, [dst, 32]
	ldp	B_l, B_h, [srcend, -32]
	stp	C_l, C_h, [dst, 48]
	ldp	C_l, C_h, [srcend, -16]
	stp	D_l, D_h, [dst, 64]
	stp	E_l, E_h, [dstend, -64]
	stp	A_l, A_h, [dstend, -48]
	stp	B_l, B_h, [dstend, -32]
	stp	C_l, C_h, [dstend, -16]
	ret

	.p2align 4
L(move_long):
	cbz	tmp1, 3f

	add	srcend, src, count
	add	dstend, dstin, count

	/* Align dstend to 16 byte alignment so that we don't cross cache line
	   boundaries on both loads and stores.  There are at least 96 bytes
	   to copy, so copy 16 bytes unaligned and then align.  The loop
	   copies 64 bytes per iteration and prefetches one iteration ahead.  */

	and	tmp1, dstend, 15
	ldp	D_l, D_h, [srcend, -16]
	sub	srcend, srcend, tmp1
	sub	count, count, tmp1
	ldp	A_l, A_h, [srcend, -16]
	stp	D_l, D_h, [dstend, -16]
	ldp	B_l, B_h, [srcend, -32]
	ldp	C_l, C_h, [srcend, -48]
	ldp	D_l, D_h, [srcend, -64]!
	sub	dstend, dstend, tmp1
	subs	count, count, 128
	b.ls	2f

	nop
1:
	stp	A_l, A_h, [dstend, -16]
	ldp	A_l, A_h, [srcend, -16]
	stp	B_l, B_h, [dstend, -32]
	ldp	B_l, B_h, [srcend, -32]
	stp	C_l, C_h, [dstend, -48]
	ldp	C_l, C_h, [srcend, -48]
	stp	D_l, D_h, [dstend, -64]!
	ldp	D_l, D_h, [srcend, -64]!
	subs	count, count, 64
	b.hi	1b

	/* Write the last full set of 64 bytes.  The remainder is at most 64
	   bytes, so it is safe to always copy 64 bytes from the start even if
	   there is just 1 byte left.  */
2:
	ldp	G_l, G_h, [src, 48]
	stp	A_l, A_h, [dstend, -16]
	ldp	A_l, A_h, [src, 32]
	stp	B_l, B_h, [dstend, -32]
	ldp	B_l, B_h, [src, 16]
	stp	C_l, C_h, [dstend, -48]
	ldp	C_l, C_h, [src]
	stp	D_l, D_h, [dstend, -64]
	stp	G_l, G_h, [dstin, 48]
	stp	A_l, A_h, [dstin, 32]
	stp	B_l, B_h, [dstin, 16]
	stp	C_l, C_h, [dstin]
3:	ret

END (MEMCPY)
libc_hidden_builtin_def (MEMCPY)

#endif
