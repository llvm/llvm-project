/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

ENTRY (__libc_pipe)
	mov	#+__NR_pipe, r3
	trapa	#0x10
	mov     r0, r3
	mov	#-12, r2
	shad	r2, r3
	not	r3, r3			// r1=0 means r0 = -1 to -4095
	tst	r3, r3			// i.e. error in linux
	bt	1f
	mov.l	r0, @r4
	mov.l	r1, @(4, r4)
	rts
	 mov	#0, r0
1:
	SYSCALL_ERROR_HANDLER
.Lpseudo_end:
	rts
	 nop
PSEUDO_END (__libc_pipe)

weak_alias (__libc_pipe, __pipe)
libc_hidden_def (__pipe)
weak_alias (__libc_pipe, pipe)
