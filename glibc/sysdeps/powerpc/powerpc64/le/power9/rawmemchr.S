/* Optimized rawmemchr implementation for PowerPC64/POWER9.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef RAWMEMCHR
# define RAWMEMCHR __rawmemchr
#endif

/* Implements the function

   int [r3] rawmemchr (void *s [r3], int c [r4])

   The implementation can load bytes past a matching byte, but only
   up to the next 16B boundary, so it never crosses a page.  */

.machine power9
ENTRY_TOCLESS (RAWMEMCHR, 4)
	CALL_MCOUNT 2

	xori	r5,r4,0xff

	mtvsrd	v18+32,r4	/* matching char in v18  */
	mtvsrd	v19+32,r5	/* non matching char in v19  */

	vspltb	v18,v18,7	/* replicate  */
	vspltb	v19,v19,7	/* replicate  */

	neg	r5,r3
	rldicl	r9,r5,0,60	/* How many bytes to get source 16B aligned?  */

	/* Align data and fill bytes not loaded with non matching char  */
	lvx	v0,0,r3
	lvsr	v1,0,r3
	vperm	v0,v19,v0,v1

	vcmpequb. v6,v0,v18	/* 0xff if byte matches, 0x00 otherwise  */
	beq	cr6,L(aligned)

	vctzlsbb r0,v6
	add	r3,r3,r0
	blr

L(aligned):
	add	r3,r3,r9

L(loop):
	lxv	v0+32,0(r3)
	vcmpequb. v6,v0,v18	/* 0xff if byte matches, 0x00 otherwise  */
	bne	cr6,L(tail1)

	lxv	v0+32,16(r3)
	vcmpequb. v6,v0,v18	/* 0xff if byte matches, 0x00 otherwise  */
	bne	cr6,L(tail2)

	lxv	v0+32,32(r3)
	vcmpequb. v6,v0,v18	/* 0xff if byte matches, 0x00 otherwise  */
	bne	cr6,L(tail3)

	lxv	v0+32,48(r3)
	vcmpequb. v6,v0,v18	/* 0xff if byte matches, 0x00 otherwise  */
	bne	cr6,L(tail4)

	addi	r3,r3,64
	b	L(loop)

L(tail1):
	vctzlsbb r0,v6
	add	r3,r3,r0
	blr

L(tail2):
	vctzlsbb r0,v6
	add	r3,r3,r0
	addi	r3,r3,16
	blr

L(tail3):
	vctzlsbb r0,v6
	add	r3,r3,r0
	addi	r3,r3,32
	blr

L(tail4):
	vctzlsbb r0,v6
	add	r3,r3,r0
	addi	r3,r3,48
	blr

END (RAWMEMCHR)
weak_alias (__rawmemchr,rawmemchr)
libc_hidden_builtin_def (__rawmemchr)
