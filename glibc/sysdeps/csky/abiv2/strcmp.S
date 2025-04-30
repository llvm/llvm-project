/* The assembly function for string compare.  C-SKY ABIV2 version.
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

ENTRY (strcmp)
	mov	a3, a0
	/* Check if the s1 addr is aligned.  */
	xor	a2, a3, a1
	andi	a2, 0x3
	bnez	a2, 7f
	andi	t1, a0, 0x3
	bnez	t1, 5f

1:
	/* If aligned, load word each time.  */
	ldw	t0, (a3, 0)
	ldw	t1, (a1, 0)
	/* If s1[i] != s2[i], goto 2f.  */
	cmpne	t0, t1
	bt	2f
	/* If s1[i] == s2[i], check if s1 or s2 is at the end.  */
	tstnbz	t0
	/* If at the end, goto 3f (finish comparing).  */
	bf	3f

	ldw	t0, (a3, 4)
	ldw	t1, (a1, 4)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	ldw	t0, (a3, 8)
	ldw	t1, (a1, 8)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	ldw	t0, (a3, 12)
	ldw	t1, (a1, 12)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	ldw	t0, (a3, 16)
	ldw	t1, (a1, 16)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	ldw	t0, (a3, 20)
	ldw	t1, (a1, 20)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	ldw	t0, (a3, 24)
	ldw	t1, (a1, 24)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	ldw	t0, (a3, 28)
	ldw	t1, (a1, 28)
	cmpne	t0, t1
	bt	2f
	tstnbz	t0
	bf	3f

	addi	a3, 32
	addi	a1, 32

	br	1b

	/* s1[i] != s2[i] in word, so we check byte 3.  */
2:
	xtrb3	a0, t0
	xtrb3	a2, t1
	subu	a0, a2
	bez	a2, 4f
	bnez	a0, 4f

	/* Check byte 2.  */
	xtrb2	a0, t0
	xtrb2	a2, t1
	subu	a0, a2
	bez	a2, 4f
	bnez	a0, 4f

	/* Check byte 1.  */
	xtrb1	a0, t0
	xtrb1	a2, t1
	subu	a0, a2
	bez	a2, 4f
	bnez	a0, 4f

	/* Check byte 0.  */
	xtrb0	a0, t0
	xtrb0	a2, t1
	subu	a0, a2

	jmp	lr
3:
	movi	a0, 0
4:
	jmp	lr

	/* Compare when s1 or s2 is not aligned.  */
5:
	subi	t1, 4
6:
	ldb	a0, (a3, 0)
	ldb	a2, (a1, 0)
	subu	a0, a2
	bnez	a0, 4b
	addi	t1, 1
	bez	a2, 4b
	addi	a1, 1
	addi	a3, 1
	bez	t1, 1b
	br	6b

7:
	ldb	a0, (a3, 0)
	addi	a3, 1
	ldb	a2, (a1, 0)
	addi	a1, 1
	subu	a0, a2
	bnez	a0, 4b
	bnez	a2, 7b
	jmp	r15
END (strcmp)

libc_hidden_def (strcmp)
.weak strcmp
