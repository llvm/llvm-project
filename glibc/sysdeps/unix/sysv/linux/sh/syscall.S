/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

ENTRY (__syscall)
	mov	r4, r3			// syscall number
	mov	r5, r4			// p1
	mov	r6, r5			// p2
	mov	r7, r6			// p3
	mov.l	@r15, r7		// p4
	mov.l	@(4,r15), r0		// p5
	mov.l	@(8,r15), r1		// p6
	mov.l	@(12,r15), r2		// p7
	trapa	#0x17
	mov     r0, r1
	mov	#-12, r2
	shad	r2, r1
	not	r1, r1			// r1=0 means r0 = -1 to -4095
	tst	r1, r1			// i.e. error in linux
	bf	.Lpseudo_end
	SYSCALL_ERROR_HANDLER
.Lpseudo_end:
	rts
	 nop

PSEUDO_END (__syscall)

weak_alias (__syscall, syscall)
