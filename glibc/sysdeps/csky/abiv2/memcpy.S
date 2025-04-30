/* The assembly function for memcpy.  C-SKY ABIV2 version.
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

ENTRY (memcpy)
	/* Test if len less than 4 bytes.  */
	mov	r12, r0
	cmplti	r2, 4
	bt	.L_copy_by_byte

	andi	r13, r0, 3
	movi	r19, 4
	/* Test if dest is not 4 bytes aligned.  */
	bnez	r13, .L_dest_not_aligned
.L_dest_aligned:
	/* If dest is aligned, then copy.  */
	zext	r18, r2, 31, 4
	/* Test if len less than 16 bytes.  */
	bez	r18, .L_len_less_16bytes
	movi	r19, 0

	LABLE_ALIGN
.L_len_larger_16bytes:
#if defined (__CSKY_VDSPV2__)
	vldx.8	vr0, (r1), r19
	PRE_BNEZAD (r18)
	addi	r1, 16
	vstx.8	vr0, (r0), r19
	addi	r0, 16
#elif defined (__csky_fpuv2__) && defined(__CK810__)
	fldd	fr4, (r1, 0)
	PRE_BNEZAD (r18)
	fstd	fr4, (r0, 0)
	fldd	fr4, (r1, 8)
	addi	r1, 16
	fstd	fr4, (r0, 8)
	addi	r0, 16
#elif defined (__CK860__)
	ldw	r3, (r1, 0)
	stw	r3, (r0, 0)
	ldw	r3, (r1, 4)
	stw	r3, (r0, 4)
	ldw	r3, (r1, 8)
	stw	r3, (r0, 8)
	ldw	r3, (r1, 12)
	addi	r1, 16
	stw	r3, (r0, 12)
	addi	r0, 16
#else
	ldw	r20, (r1, 0)
	ldw	r21, (r1, 4)
	ldw	r22, (r1, 8)
	ldw	r23, (r1, 12)
	stw	r20, (r0, 0)
	stw	r21, (r0, 4)
	stw	r22, (r0, 8)
	stw	r23, (r0, 12)
	PRE_BNEZAD (r18)
	addi	r1, 16
	addi	r0, 16
#endif
	BNEZAD (r18, .L_len_larger_16bytes)

.L_len_less_16bytes:
	zext	r18, r2, 3, 2
	bez	r18, .L_copy_by_byte
.L_len_less_16bytes_loop:
	ldw	r3, (r1, 0)
	PRE_BNEZAD (r18)
	addi	r1, 4
	stw	r3, (r0, 0)
	addi	r0, 4
	BNEZAD (r18, .L_len_less_16bytes_loop)

	/* Test if len less than 4 bytes.  */
.L_copy_by_byte:
	zext	r18, r2, 1, 0
	bez	r18, .L_return
.L_copy_by_byte_loop:
	ldb	r3, (r1, 0)
	PRE_BNEZAD (r18)
	addi	r1, 1
	stb	r3, (r0, 0)
	addi	r0, 1
	BNEZAD (r18, .L_copy_by_byte_loop)

.L_return:
	mov	r0, r12
	rts

	/* If dest is not aligned, just copying some bytes makes the dest
	   align.  */

.L_dest_not_aligned:
	sub	r13, r19, r13
	mov	r19, r13
.L_dest_not_aligned_loop:
	/* Makes the dest align.  */
	ldb	r3, (r1, 0)
	PRE_BNEZAD (r13)
	addi	r1, 1
	stb	r3, (r0, 0)
	addi	r0, 1
	BNEZAD (r13, .L_dest_not_aligned_loop)
	sub	r2, r19
	cmplti	r2, 4
	bt	.L_copy_by_byte
	/* Check whether the src is aligned.  */
	br	.L_dest_aligned
END (memcpy)

libc_hidden_builtin_def (memcpy)
.weak memcpy


ENTRY (memmove)
	subu	r3, r0, r1
	cmphs	r3, r2
	bt	memcpy

	mov	r12, r0
	addu	r0, r0, r2
	addu	r1, r1, r2

	/* Test if len less than 4 bytes.  */
	cmplti	r2, 4
	bt	.L_copy_by_byte_m

	andi	r13, r0, 3
	/* Test if dest is not 4 bytes aligned.  */
	bnez	r13, .L_dest_not_aligned_m
.L_dest_aligned_m:
	/* If dest is aligned, then copy.  */
	zext	r18, r2, 31, 4
	/* Test if len less than 16 bytes.  */
	bez	r18, .L_len_less_16bytes_m
	movi	r19, 0

	/* len > 16 bytes */
	LABLE_ALIGN
.L_len_larger_16bytes_m:
	subi	r1, 16
	subi	r0, 16
#if defined (__CSKY_VDSPV2__)
	vldx.8	vr0, (r1), r19
	PRE_BNEZAD (r18)
	vstx.8	vr0, (r0), r19
#elif defined (__csky_fpuv2__) && defined(__CK810__)
	fldd	fr4, (r1, 8)
	PRE_BNEZAD (r18)
	fstd	fr4, (r0, 8)
	fldd	fr4, (r1, 0)
	fstd	fr4, (r0, 0)
#elif defined (__CK860__)
	ldw	r3, (r1, 12)
	stw	r3, (r0, 12)
	ldw	r3, (r1, 8)
	stw	r3, (r0, 8)
	ldw	r3, (r1, 4)
	stw	r3, (r0, 4)
	ldw	r3, (r1, 0)
	stw	r3, (r0, 0)
#else
	ldw	r20, (r1, 0)
	ldw	r21, (r1, 4)
	ldw	r22, (r1, 8)
	ldw	r23, (r1, 12)
	stw	r20, (r0, 0)
	stw	r21, (r0, 4)
	stw	r22, (r0, 8)
	stw	r23, (r0, 12)
	PRE_BNEZAD (r18)
#endif
	BNEZAD (r18, .L_len_larger_16bytes_m)

.L_len_less_16bytes_m:
	zext	r18, r2, 3, 2
	bez	r18, .L_copy_by_byte_m
.L_len_less_16bytes_loop_m:
	subi	r1, 4
	subi	r0, 4
	ldw	r3, (r1, 0)
	PRE_BNEZAD (r18)
	stw	r3, (r0, 0)
	BNEZAD (r18, .L_len_less_16bytes_loop_m)

	/* Test if len less than 4 bytes.  */
.L_copy_by_byte_m:
	zext	r18, r2, 1, 0
	bez	r18, .L_return_m
.L_copy_by_byte_loop_m:
	subi	r1, 1
	subi	r0, 1
	ldb	r3, (r1, 0)
	PRE_BNEZAD (r18)
	stb	r3, (r0, 0)
	BNEZAD (r18, .L_copy_by_byte_loop_m)

.L_return_m:
	mov	r0, r12
	rts

	/* If dest is not aligned, just copying some bytes makes the dest
	   align.  */
.L_dest_not_aligned_m:
	sub	r2, r13
.L_dest_not_aligned_loop_m:
	subi	r1, 1
	subi	r0, 1
	/* Makes the dest align.  */
	ldb	r3, (r1, 0)
	PRE_BNEZAD (r13)
	stb	r3, (r0, 0)
	BNEZAD (r13, .L_dest_not_aligned_loop_m)
	cmplti	r2, 4
	bt	.L_copy_by_byte_m
	/* Check whether the src is aligned.  */
	br	.L_dest_aligned_m
END (memmove)

libc_hidden_builtin_def (memmove)
.weak memmove
