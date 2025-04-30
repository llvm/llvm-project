/* A Thunderx2 Optimized memcpy implementation for AARCH64.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

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
#define tmp2	x6
#define tmp3	x7
#define tmp3w   w7
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

#define A_q	q0
#define B_q	q1
#define C_q	q2
#define D_q	q3
#define E_q	q4
#define F_q	q5
#define G_q	q6
#define H_q	q7
#define I_q	q16
#define J_q	q17

#define A_v	v0
#define B_v	v1
#define C_v	v2
#define D_v	v3
#define E_v	v4
#define F_v	v5
#define G_v	v6
#define H_v	v7
#define I_v	v16
#define J_v	v17

#ifndef MEMMOVE
# define MEMMOVE memmove
#endif
#ifndef MEMCPY
# define MEMCPY memcpy
#endif

#if IS_IN (libc)

#undef MEMCPY
#define MEMCPY __memcpy_thunderx2
#undef MEMMOVE
#define MEMMOVE __memmove_thunderx2


/* Overlapping large forward memmoves use a loop that copies backwards.
   Otherwise memcpy is used. Small moves branch to memcopy16 directly.
   The longer memcpy cases fall through to the memcpy head.
*/

ENTRY_ALIGN (MEMMOVE, 6)

	PTR_ARG (0)
	PTR_ARG (1)
	SIZE_ARG (2)

	add	srcend, src, count
	cmp	count, 16
	b.ls	L(memcopy16)
	sub	tmp1, dstin, src
	cmp	count, 96
	ccmp	tmp1, count, 2, hi
	b.lo	L(move_long)

END (MEMMOVE)
libc_hidden_builtin_def (MEMMOVE)


/* Copies are split into 3 main cases: small copies of up to 16 bytes,
   medium copies of 17..96 bytes which are fully unrolled. Large copies
   of more than 96 bytes align the destination and use load-and-merge
   approach in the case src and dst addresses are unaligned not evenly,
   so that, actual loads and stores are always aligned.
   Large copies use the loops processing 64 bytes per iteration for
   unaligned case and 128 bytes per iteration for aligned ones.
*/

#define MEMCPY_PREFETCH_LDR 640

	.p2align 4
ENTRY (MEMCPY)

	PTR_ARG (0)
	PTR_ARG (1)
	SIZE_ARG (2)

	add	srcend, src, count
	cmp	count, 16
	b.ls	L(memcopy16)
	ldr	A_q, [src], #16
	add	dstend, dstin, count
	and	tmp1, src, 15
	cmp	count, 96
	b.hi	L(memcopy_long)

	/* Medium copies: 17..96 bytes.  */
	ldr	E_q, [srcend, -16]
	cmp	count, 64
	b.gt	L(memcpy_copy96)
	cmp	count, 48
	b.le	L(bytes_17_to_48)
	/* 49..64 bytes */
	ldp	B_q, C_q, [src]
	str	E_q, [dstend, -16]
	stp	A_q, B_q, [dstin]
	str	C_q, [dstin, 32]
	ret

L(bytes_17_to_48):
	/* 17..48 bytes*/
	cmp	count, 32
	b.gt	L(bytes_32_to_48)
	/* 17..32 bytes*/
	str	A_q, [dstin]
	str	E_q, [dstend, -16]
	ret

L(bytes_32_to_48):
	/* 32..48 */
	ldr	B_q, [src]
	str	A_q, [dstin]
	str	E_q, [dstend, -16]
	str	B_q, [dstin, 16]
	ret

	.p2align 4
	/* Small copies: 0..16 bytes.  */
L(memcopy16):
	cmp	count, 8
	b.lo	L(bytes_0_to_8)
	ldr	A_l, [src]
	ldr	A_h, [srcend, -8]
	add	dstend, dstin, count
	str	A_l, [dstin]
	str	A_h, [dstend, -8]
	ret
	.p2align 4

L(bytes_0_to_8):
	tbz	count, 2, L(bytes_0_to_3)
	ldr	A_lw, [src]
	ldr	A_hw, [srcend, -4]
	add	dstend, dstin, count
	str	A_lw, [dstin]
	str	A_hw, [dstend, -4]
	ret

	/* Copy 0..3 bytes.  Use a branchless sequence that copies the same
	   byte 3 times if count==1, or the 2nd byte twice if count==2.  */
L(bytes_0_to_3):
	cbz	count, 1f
	lsr	tmp1, count, 1
	ldrb	A_lw, [src]
	ldrb	A_hw, [srcend, -1]
	add	dstend, dstin, count
	ldrb	B_lw, [src, tmp1]
	strb	B_lw, [dstin, tmp1]
	strb	A_hw, [dstend, -1]
	strb	A_lw, [dstin]
1:
	ret

	.p2align 4

L(memcpy_copy96):
	/* Copying 65..96 bytes. A_q (first 16 bytes) and
	   E_q(last 16 bytes) are already loaded. The size
	   is large enough to benefit from aligned loads */
	bic	src, src, 15
	ldp	B_q, C_q, [src]
	/* Loaded 64 bytes, second 16-bytes chunk can be
	   overlapping with the first chunk by tmp1 bytes.
	   Stored 16 bytes. */
	sub	dst, dstin, tmp1
	add	count, count, tmp1
	/* The range of count being [65..96] becomes [65..111]
	   after tmp [0..15] gets added to it,
	   count now is <bytes-left-to-load>+48 */
	cmp	count, 80
	b.gt	L(copy96_medium)
	ldr	D_q, [src, 32]
	stp	B_q, C_q, [dst, 16]
	str	D_q, [dst, 48]
	str	A_q, [dstin]
	str	E_q, [dstend, -16]
	ret

	.p2align 4
L(copy96_medium):
	ldp	D_q, G_q, [src, 32]
	cmp	count, 96
	b.gt	L(copy96_large)
	stp	B_q, C_q, [dst, 16]
	stp	D_q, G_q, [dst, 48]
	str	A_q, [dstin]
	str	E_q, [dstend, -16]
	ret

L(copy96_large):
	ldr	F_q, [src, 64]
	str	B_q, [dst, 16]
	stp	C_q, D_q, [dst, 32]
	stp	G_q, F_q, [dst, 64]
	str	A_q, [dstin]
	str	E_q, [dstend, -16]
	ret

	.p2align 4
L(memcopy_long):
	bic	src, src, 15
	ldp	B_q, C_q, [src], #32
	sub	dst, dstin, tmp1
	add	count, count, tmp1
	add	dst, dst, 16
	and	tmp1, dst, 15
	ldp	D_q, E_q, [src], #32
	str	A_q, [dstin]

	/* Already loaded 64+16 bytes. Check if at
	   least 64 more bytes left */
	subs	count, count, 64+64+16
	b.lt	L(loop128_exit0)
	cmp	count, MEMCPY_PREFETCH_LDR + 64 + 32
	b.lt	L(loop128)
	cbnz	tmp1, L(dst_unaligned)
	sub	count, count, MEMCPY_PREFETCH_LDR + 64 + 32

	.p2align 4

L(loop128_prefetch):
	prfm	pldl1strm, [src, MEMCPY_PREFETCH_LDR]
	ldp	F_q, G_q, [src], #32
	stp	B_q, C_q, [dst], #32
	ldp	H_q, I_q, [src], #32
	prfm	pldl1strm, [src, MEMCPY_PREFETCH_LDR]
	ldp	B_q, C_q, [src], #32
	stp	D_q, E_q, [dst], #32
	ldp	D_q, E_q, [src], #32
	stp	F_q, G_q, [dst], #32
	stp	H_q, I_q, [dst], #32
	subs	count, count, 128
	b.ge	L(loop128_prefetch)

	add	count, count, MEMCPY_PREFETCH_LDR + 64 + 32
	.p2align 4
L(loop128):
	ldp	F_q, G_q, [src], #32
	ldp	H_q, I_q, [src], #32
	stp	B_q, C_q, [dst], #32
	stp	D_q, E_q, [dst], #32
	subs	count, count, 64
	b.lt	L(loop128_exit1)
	ldp	B_q, C_q, [src], #32
	ldp	D_q, E_q, [src], #32
	stp	F_q, G_q, [dst], #32
	stp	H_q, I_q, [dst], #32
	subs	count, count, 64
	b.ge	L(loop128)
L(loop128_exit0):
	ldp	F_q, G_q, [srcend, -64]
	ldp	H_q, I_q, [srcend, -32]
	stp	B_q, C_q, [dst], #32
	stp	D_q, E_q, [dst]
	stp	F_q, G_q, [dstend, -64]
	stp	H_q, I_q, [dstend, -32]
	ret
L(loop128_exit1):
	ldp	B_q, C_q, [srcend, -64]
	ldp	D_q, E_q, [srcend, -32]
	stp	F_q, G_q, [dst], #32
	stp	H_q, I_q, [dst]
	stp	B_q, C_q, [dstend, -64]
	stp	D_q, E_q, [dstend, -32]
	ret

L(dst_unaligned_tail):
	ldp	C_q, D_q, [srcend, -64]
	ldp	E_q, F_q, [srcend, -32]
	stp	A_q, B_q, [dst], #32
	stp	H_q, I_q, [dst], #16
	str	G_q, [dst, tmp1]
	stp	C_q, D_q, [dstend, -64]
	stp	E_q, F_q, [dstend, -32]
	ret

L(dst_unaligned):
	/* For the unaligned store case the code loads two
	   aligned chunks and then merges them using ext
	   instruction. This can be up to 30% faster than
	   the the simple unaligned store access.

	   Current state: tmp1 = dst % 16; C_q, D_q, E_q
	   contains data yet to be stored. src and dst points
	   to next-to-be-processed data. A_q, B_q contains
	   data already stored before, count = bytes left to
	   be load decremented by 64.

	   The control is passed here if at least 64 bytes left
	   to be loaded. The code does two aligned loads and then
	   extracts (16-tmp1) bytes from the first register and
	   tmp1 bytes from the next register forming the value
	   for the aligned store.

	   As ext instruction can only have it's index encoded
	   as immediate. 15 code chunks process each possible
	   index value. Computed goto is used to reach the
	   required code. */

	/* Store the 16 bytes to dst and align dst for further
	   operations, several bytes will be stored at this
	   address once more */

	ldp	F_q, G_q, [src], #32
	stp	B_q, C_q, [dst], #32
	bic	dst, dst, 15
	sub	count, count, 32
	adrp	tmp2, L(ext_table)
	add	tmp2, tmp2, :lo12:L(ext_table)
	add	tmp2, tmp2, tmp1, LSL #2
	ldr	tmp3w, [tmp2]
	add	tmp2, tmp2, tmp3w, SXTW
	br	tmp2

.p2align 4
	/* to make the loop in each chunk 16-bytes aligned */
	nop
#define EXT_CHUNK(shft) \
L(ext_size_ ## shft):;\
	ext     A_v.16b, C_v.16b, D_v.16b, 16-shft;\
	ext     B_v.16b, D_v.16b, E_v.16b, 16-shft;\
	ext     H_v.16b, E_v.16b, F_v.16b, 16-shft;\
1:;\
	stp     A_q, B_q, [dst], #32;\
	prfm    pldl1strm, [src, MEMCPY_PREFETCH_LDR];\
	ldp     C_q, D_q, [src], #32;\
	ext     I_v.16b, F_v.16b, G_v.16b, 16-shft;\
	stp     H_q, I_q, [dst], #32;\
	ext     A_v.16b, G_v.16b, C_v.16b, 16-shft;\
	ext     B_v.16b, C_v.16b, D_v.16b, 16-shft;\
	ldp     F_q, G_q, [src], #32;\
	ext     H_v.16b, D_v.16b, F_v.16b, 16-shft;\
	subs    count, count, 64;\
	b.ge    1b;\
2:;\
	ext     I_v.16b, F_v.16b, G_v.16b, 16-shft;\
	b	L(dst_unaligned_tail);

EXT_CHUNK(1)
EXT_CHUNK(2)
EXT_CHUNK(3)
EXT_CHUNK(4)
EXT_CHUNK(5)
EXT_CHUNK(6)
EXT_CHUNK(7)
EXT_CHUNK(8)
EXT_CHUNK(9)
EXT_CHUNK(10)
EXT_CHUNK(11)
EXT_CHUNK(12)
EXT_CHUNK(13)
EXT_CHUNK(14)
EXT_CHUNK(15)

L(move_long):
	.p2align 4
1:
	cbz	tmp1, 3f

	add	srcend, src, count
	add	dstend, dstin, count

	and	tmp1, srcend, 15
	ldr	D_q, [srcend, -16]
	sub	srcend, srcend, tmp1
	sub	count, count, tmp1
	ldp	A_q, B_q, [srcend, -32]
	str	D_q, [dstend, -16]
	ldp	C_q, D_q, [srcend, -64]!
	sub	dstend, dstend, tmp1
	subs	count, count, 128
	b.ls	2f

	.p2align 4
1:
	subs	count, count, 64
	stp	A_q, B_q, [dstend, -32]
	ldp	A_q, B_q, [srcend, -32]
	stp	C_q, D_q, [dstend, -64]!
	ldp	C_q, D_q, [srcend, -64]!
	b.hi	1b

	/* Write the last full set of 64 bytes.  The remainder is at most 64
	   bytes, so it is safe to always copy 64 bytes from the start even if
	   there is just 1 byte left.  */
2:
	ldp	E_q, F_q, [src, 32]
	ldp	G_q, H_q, [src]
	stp	A_q, B_q, [dstend, -32]
	stp	C_q, D_q, [dstend, -64]
	stp	E_q, F_q, [dstin, 32]
	stp	G_q, H_q, [dstin]
3:	ret


END (MEMCPY)
	.section	.rodata
	.p2align	4

L(ext_table):
	/* The first entry is for the alignment of 0 and is never
	   actually used (could be any value).  */
	.word	0
	.word	L(ext_size_1) -.
	.word	L(ext_size_2) -.
	.word	L(ext_size_3) -.
	.word	L(ext_size_4) -.
	.word	L(ext_size_5) -.
	.word	L(ext_size_6) -.
	.word	L(ext_size_7) -.
	.word	L(ext_size_8) -.
	.word	L(ext_size_9) -.
	.word	L(ext_size_10) -.
	.word	L(ext_size_11) -.
	.word	L(ext_size_12) -.
	.word	L(ext_size_13) -.
	.word	L(ext_size_14) -.
	.word	L(ext_size_15) -.

libc_hidden_builtin_def (MEMCPY)
#endif
