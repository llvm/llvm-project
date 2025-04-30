/* The assembly function for string length.  C-SKY ABIV2 version.
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

ENTRY (strlen)
	/* Check if the start addr is aligned.  */
	mov	r3, r0
	andi	r1, r0, 3
	movi	r2, 4
	movi	r0, 0
	bnez	r1, .L_start_not_aligned

	LABLE_ALIGN
.L_start_addr_aligned:
	/* Check if all the bytes in the word are not zero.  */
	ldw	r1, (r3)
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 4)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 8)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 12)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 16)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 20)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 24)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	ldw	r1, (r3, 28)
	addi	r0, 4
	tstnbz	r1
	bf	.L_string_tail

	addi	r0, 4
	addi	r3, 32
	br	.L_start_addr_aligned

.L_string_tail:
	xtrb3	r3, r1
	bez	r3, .L_return
	addi	r0, 1
	xtrb2	r3, r1
	bez	r3, .L_return
	addi	r0, 1
	xtrb1	r3, r1
	bez	r3, .L_return
	addi	r0, 1

.L_return:
	rts

.L_start_not_aligned:
	sub	r2, r2, r1
.L_start_not_aligned_loop:
	ldb	r1, (r3)
	PRE_BNEZAD (r2)
	addi	r3, 1
	bez	r1, .L_return
	addi	r0, 1
	BNEZAD (r2, .L_start_not_aligned_loop)
	br	.L_start_addr_aligned
END (strlen)

libc_hidden_def (strlen)
.weak strlen
