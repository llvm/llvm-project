/* ffsll -- find first set bit in a long long, from least significant end.
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

	.syntax unified
	.text

ENTRY (ffsll)
	@ If low part is 0, operate on the high part.  Ensure that the
	@ word on which we operate is in r0.  Set r2 to the bit offset
	@ of the word being considered.  Set the flags for the word
	@ being operated on.
#ifdef __ARMEL__
	cmp	r0, #0
	itee	ne
	movne	r2, #0
	moveq	r2, #32
	movseq	r0, r1
#else
	cmp	r1, #0
	ittee	ne
	movne	r2, #0
	movne	r0, r1
	moveq	r2, #32
	cmpeq	r0, #0
#endif
	@ Perform the ffs on r0.
	rbit	r0, r0
	ittt	ne
	clzne	r0, r0
	addne	r2, r2, #1
	addne	r0, r0, r2
	bx	lr
END (ffsll)
