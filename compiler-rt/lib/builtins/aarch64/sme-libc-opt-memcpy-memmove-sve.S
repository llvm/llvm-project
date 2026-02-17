//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains assembly-optimized implementations of Scalable Matrix
/// Extension (SME) compatible memcpy and memmove functions.
///
/// These implementations depend on unaligned access support.
///
/// Routines taken from libc/AOR_v20.02/string/aarch64.
///
//===----------------------------------------------------------------------===//

#include "../assembly.h"

#if defined(__ARM_FEATURE_SVE)

//
//  __arm_sc_memcpy / __arm_sc_memmove
//

#define dstin	x0
#define src	x1
#define count	x2
#define dst	x3
#define srcend	x4
#define dstend	x5
#define tmp1	x6
#define vlen	x6

#define A_q	q0
#define B_q	q1
#define C_q	q2
#define D_q	q3
#define E_q	q4
#define F_q	q5
#define G_q	q6
#define H_q	q7

/* This implementation handles overlaps and supports both memcpy and memmove
   from a single entry point.  It uses unaligned accesses and branchless
   sequences to keep the code small, simple and improve performance.
   SVE vectors are used to speedup small copies.

   Copies are split into 3 main cases: small copies of up to 32 bytes, medium
   copies of up to 128 bytes, and large copies.  The overhead of the overlap
   check is negligible since it is only required for large copies.

   Large copies use a software pipelined loop processing 64 bytes per iteration.
   The source pointer is 16-byte aligned to minimize unaligned accesses.
   The loop tail is handled by always copying 64 bytes from the end.
*/

DEFINE_COMPILERRT_FUNCTION(__arm_sc_memcpy)
	cmp	count, 128
	b.hi	3f  // copy_long
	cntb	vlen
	cmp	count, vlen, lsl 1
	b.hi	0f  // copy32_128

	whilelo p0.b, xzr, count
	whilelo p1.b, vlen, count
	ld1b	z0.b, p0/z, [src, 0, mul vl]
	ld1b	z1.b, p1/z, [src, 1, mul vl]
	st1b	z0.b, p0, [dstin, 0, mul vl]
	st1b	z1.b, p1, [dstin, 1, mul vl]
	ret

	/* Medium copies: 33..128 bytes.  */
0:  // copy32_128
	add	srcend, src, count
	add	dstend, dstin, count
	ldp	A_q, B_q, [src]
	ldp	C_q, D_q, [srcend, -32]
	cmp	count, 64
	b.hi	1f  // copy128
	stp	A_q, B_q, [dstin]
	stp	C_q, D_q, [dstend, -32]
	ret

	/* Copy 65..128 bytes.  */
1:  // copy128
	ldp	E_q, F_q, [src, 32]
	cmp	count, 96
	b.ls	2f  // copy96
	ldp	G_q, H_q, [srcend, -64]
	stp	G_q, H_q, [dstend, -64]
2:  // copy96
	stp	A_q, B_q, [dstin]
	stp	E_q, F_q, [dstin, 32]
	stp	C_q, D_q, [dstend, -32]
	ret

	/* Copy more than 128 bytes.  */
3:  // copy_long
	add	srcend, src, count
	add	dstend, dstin, count

	/* Use backwards copy if there is an overlap.  */
	sub	tmp1, dstin, src
	cmp	tmp1, count
	b.lo	6f  // copy_long_backwards

	/* Copy 16 bytes and then align src to 16-byte alignment.  */
	ldr	D_q, [src]
	and	tmp1, src, 15
	bic	src, src, 15
	sub	dst, dstin, tmp1
	add	count, count, tmp1	/* Count is now 16 too large.  */
	ldp	A_q, B_q, [src, 16]
	str	D_q, [dstin]
	ldp	C_q, D_q, [src, 48]
	subs	count, count, 128 + 16	/* Test and readjust count.  */
	b.ls	5f  // copy64_from_end
4:  // loop64
	stp	A_q, B_q, [dst, 16]
	ldp	A_q, B_q, [src, 80]
	stp	C_q, D_q, [dst, 48]
	ldp	C_q, D_q, [src, 112]
	add	src, src, 64
	add	dst, dst, 64
	subs	count, count, 64
	b.hi	4b  // loop64

	/* Write the last iteration and copy 64 bytes from the end.  */
5:  // copy64_from_end
	ldp	E_q, F_q, [srcend, -64]
	stp	A_q, B_q, [dst, 16]
	ldp	A_q, B_q, [srcend, -32]
	stp	C_q, D_q, [dst, 48]
	stp	E_q, F_q, [dstend, -64]
	stp	A_q, B_q, [dstend, -32]
	ret

	/* Large backwards copy for overlapping copies.
	   Copy 16 bytes and then align srcend to 16-byte alignment.  */
6:  // copy_long_backwards
	cbz	tmp1, 9f  // return
	ldr	D_q, [srcend, -16]
	and	tmp1, srcend, 15
	bic	srcend, srcend, 15
	sub	count, count, tmp1
	ldp	A_q, B_q, [srcend, -32]
	str	D_q, [dstend, -16]
	ldp	C_q, D_q, [srcend, -64]
	sub	dstend, dstend, tmp1
	subs	count, count, 128
	b.ls	8f  // copy64_from_start

7:  // loop64_backwards
	str	B_q, [dstend, -16]
	str	A_q, [dstend, -32]
	ldp	A_q, B_q, [srcend, -96]
	str	D_q, [dstend, -48]
	str	C_q, [dstend, -64]!
	ldp	C_q, D_q, [srcend, -128]
	sub	srcend, srcend, 64
	subs	count, count, 64
	b.hi	7b  // loop64_backwards

	/* Write the last iteration and copy 64 bytes from the start.  */
8:  // copy64_from_start
	ldp	E_q, F_q, [src, 32]
	stp	A_q, B_q, [dstend, -32]
	ldp	A_q, B_q, [src]
	stp	C_q, D_q, [dstend, -64]
	stp	E_q, F_q, [dstin, 32]
	stp	A_q, B_q, [dstin]
9: // return
	ret
END_COMPILERRT_FUNCTION (__arm_sc_memcpy)

DEFINE_COMPILERRT_FUNCTION_ALIAS(__arm_sc_memmove, __arm_sc_memcpy)

#endif // defined(__ARM_FEATURE_SVE)
