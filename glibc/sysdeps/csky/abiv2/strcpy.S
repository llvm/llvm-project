/* The assembly function for string copy.  C-SKY ABIV2 version.
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

ENTRY (strcpy)
	mov	a3, a0
	/* Check if the src addr is aligned.  */
	andi	t0, a1, 3
	bnez	t0, 11f
1:
	/* Check if all the bytes in the word are not zero.  */
	ldw	a2, (a1)
	tstnbz	a2
	bf	9f
	stw	a2, (a3)

	ldw	a2, (a1, 4)
	tstnbz	a2
	bf	2f
	stw	a2, (a3, 4)

	ldw	a2, (a1, 8)
	tstnbz	a2
	bf	3f
	stw	a2, (a3, 8)

	ldw	a2, (a1, 12)
	tstnbz	a2
	bf	4f
	stw	a2, (a3, 12)

	ldw	a2, (a1, 16)
	tstnbz	a2
	bf	5f
	stw	a2, (a3, 16)

	ldw	a2, (a1, 20)
	tstnbz	a2
	bf	6f
	stw	a2, (a3, 20)

	ldw	a2, (a1, 24)
	tstnbz	a2
	bf	7f
	stw	a2, (a3, 24)

	ldw	a2, (a1, 28)
	tstnbz	a2
	bf	8f
	stw	a2, (a3, 28)

	addi	a3, 32
	addi	a1, 32
	br	1b

2:
	addi	a3, 4
	br	9f

3:
	addi	a3, 8
	br	9f

4:
	addi	a3, 12
	br	9f

5:
	addi	a3, 16
	br	9f

6:
	addi	a3, 20
	br	9f

7:
	addi	a3, 24
	br	9f

8:
	addi	a3, 28
9:
	xtrb3	t0, a2
	st.b	t0, (a3)
	bez	t0, 10f
	xtrb2	t0, a2
	st.b	t0, (a3, 1)
	bez	t0, 10f
	xtrb1	t0, a2
	st.b	t0, (a3, 2)
	bez	t0, 10f
	stw	a2, (a3)
10:
	jmp	lr

11:
	subi	t0, 4
12:
	ld.b	a2, (a1)
	st.b	a2, (a3)
	bez	a2, 10b
	addi	t0, 1
	addi	a1, a1, 1
	addi	a3, a3, 1
	bnez	t0, 12b
	br	1b

END (strcpy)

libc_hidden_def (strcpy)
.weak strcpy
