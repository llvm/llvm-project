/* The assembly function for memory compare.  C-SKY ABIV2 version.
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

ENTRY (memcmp)
	/* Test if len less than 4 bytes.  */
	mov	r3, r0
	movi	r0, 0
	mov	r12, r4
	cmplti	r2, 4
	jbt	.L_compare_by_byte

	andi	r13, r0, 3
	movi	r19, 4
	/* Test if s1 is not 4 bytes aligned.  */
	bnez	r13, .L_s1_not_aligned

	LABLE_ALIGN
.L_s1_aligned:
	/* If dest is aligned, then copy.  */
	zext	r18, r2, 31, 4
	/* Test if len less than 16 bytes.  */
	bez	r18, .L_compare_by_word

.L_compare_by_4word:
	/* If aligned, load word each time.  */
	ldw	r20, (r3, 0)
	ldw	r21, (r1, 0)
	/* If s1[i] != s2[i], goto .L_byte_check.  */
	cmpne	r20, r21
	bt	.L_byte_check

	ldw	r20, (r3, 4)
	ldw	r21, (r1, 4)
	cmpne	r20, r21
	bt	.L_byte_check

	ldw	r20, (r3, 8)
	ldw	r21, (r1, 8)
	cmpne	r20, r21
	bt	.L_byte_check

	ldw	r20, (r3, 12)
	ldw	r21, (r1, 12)
	cmpne	r20, r21
	bt	.L_byte_check

	PRE_BNEZAD (r18)
	addi	a3, 16
	addi	a1, 16

	BNEZAD (r18, .L_compare_by_4word)

.L_compare_by_word:
	zext	r18, r2, 3, 2
	bez	r18, .L_compare_by_byte
.L_compare_by_word_loop:
	ldw	r20, (r3, 0)
	ldw	r21, (r1, 0)
	addi	r3, 4
	PRE_BNEZAD (r18)
	cmpne	r20, r21
	addi	r1, 4
	bt	.L_byte_check
	BNEZAD (r18, .L_compare_by_word_loop)

.L_compare_by_byte:
	zext	r18, r2, 1, 0
	bez	r18, .L_return
.L_compare_by_byte_loop:
	ldb	r0, (r3, 0)
	ldb	r4, (r1, 0)
	addi	r3, 1
	subu	r0, r4
	PRE_BNEZAD (r18)
	addi	r1, 1
	bnez	r0, .L_return
	BNEZAD (r18, .L_compare_by_byte_loop)

.L_return:
	mov	r4, r12
	rts

	/* s1[i] != s2[i] in word, so we check byte 3.  */
.L_byte_check:
	xtrb3	r0, r20
	xtrb3	r2, r21
	subu	r0, r2
	bnez	r0, .L_return

	/* check byte 2 */
	xtrb2	r0, r20
	xtrb2	r2, r21
	subu	r0, r2
	bnez	r0, .L_return

	/* check byte 1 */
	xtrb1	r0, r20
	xtrb1	r2, r21
	subu	r0, r2
	bnez	r0, .L_return

	/* check byte 0 */
	xtrb0	r0, r20
	xtrb0	r2, r21
	subu	r0, r2
	br	.L_return

	/* Compare when s1 is not aligned.  */
.L_s1_not_aligned:
	sub	r13, r19, r13
	sub	r2, r13
.L_s1_not_aligned_loop:
	ldb	r0, (r3, 0)
	ldb	r4, (r1, 0)
	addi	r3, 1
	subu	r0, r4
	PRE_BNEZAD (r13)
	addi	r1, 1
	bnez	r0, .L_return
	BNEZAD (r13, .L_s1_not_aligned_loop)
	br	.L_s1_aligned
END (memcmp)
weak_alias (memcmp, bcmp)
libc_hidden_def (memcmp)
.weak memcmp
