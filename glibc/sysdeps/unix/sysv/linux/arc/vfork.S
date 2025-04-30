/* vfork for ARC Linux.
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

#include <sys/syscall.h>
#include <sysdep.h>
#include <tcb-offsets.h>
#define _SIGNAL_H
#include <bits/signum-arch.h>       /* For SIGCHLD */

#define CLONE_VM		0x00000100
#define CLONE_VFORK		0x00004000
#define CLONE_FLAGS_FOR_VFORK	(CLONE_VM|CLONE_VFORK|SIGCHLD)

ENTRY (__vfork)
	mov	r0, CLONE_FLAGS_FOR_VFORK
	mov_s	r1, sp
	mov	r8, __NR_clone
	ARC_TRAP_INSN

	cmp	r0, 0
	jge	[blink]	; child continues

	b   __syscall_error
PSEUDO_END (__vfork)
libc_hidden_def (__vfork)

weak_alias (__vfork, vfork)
