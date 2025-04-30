/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <philb@gnu.org>

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

/* Thumb requires excessive IT insns here.  */
#define NO_THUMB
#include <sysdep.h>

	.text
	.syntax unified

/* void *memset (dstpp, c, len) */

ENTRY(memset)
	mov	r3, r0
	cmp	r2, #8
	bcc	2f		@ less than 8 bytes to move

1:
	tst	r3, #3		@ aligned yet?
	strbne	r1, [r3], #1
	subne	r2, r2, #1
	bne	1b

	and	r1, r1, #255	@ clear any sign bits
	orr	r1, r1, r1, lsl $8
	orr	r1, r1, r1, lsl $16
	mov	ip, r1

1:
	subs	r2, r2, #8
	stmiacs	r3!, {r1, ip}	@ store up to 32 bytes per loop iteration
	subscs	r2, r2, #8
	stmiacs	r3!, {r1, ip}
	subscs	r2, r2, #8
	stmiacs	r3!, {r1, ip}
	subscs	r2, r2, #8
	stmiacs	r3!, {r1, ip}
	bcs	1b

	and	r2, r2, #7
2:
	subs	r2, r2, #1	@ store up to 4 bytes per loop iteration
	strbcs	r1, [r3], #1
	subscs	r2, r2, #1
	strbcs	r1, [r3], #1
	subscs	r2, r2, #1
	strbcs	r1, [r3], #1
	subscs	r2, r2, #1
	strbcs	r1, [r3], #1
	bcs	2b

	DO_RET(lr)
END(memset)
libc_hidden_builtin_def (memset)
