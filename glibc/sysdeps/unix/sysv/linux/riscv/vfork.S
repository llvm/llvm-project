/* vfork for Linux, RISC-V version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* vfork() is just a special case of clone().  */

#include <sys/asm.h>
#include <sysdep.h>
#define __ASSEMBLY__
#include <asm/signal.h>

#define CLONE_VM      0x00000100 /* Set if VM shared between processes.  */
#define CLONE_VFORK   0x00004000 /* Set if the parent wants the child to
				    wake it up on mm_release.  */

	.text
LEAF (__libc_vfork)

	li	a0, (CLONE_VFORK | CLONE_VM | SIGCHLD)
	mv	a1, sp

	li	a7, __NR_clone
	scall

	bltz	a0, 1f
	ret

1:	j		__syscall_error
END (__libc_vfork)

weak_alias (__libc_vfork, vfork)
strong_alias (__libc_vfork, __vfork)
libc_hidden_def (__vfork)
