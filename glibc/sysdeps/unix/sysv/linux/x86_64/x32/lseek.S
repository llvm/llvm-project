/* The lseek system call with 64-bit offset.  Linux/x32 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Return -1LL in a full 64 bits.  */
#undef SYSCALL_ERROR_HANDLER
#define SYSCALL_ERROR_HANDLER	\
0:				\
	SYSCALL_SET_ERRNO;	\
	orq $-1, %rax;		\
	ret;

/* Always use our own error handler.  */
#undef SYSCALL_ERROR_LABEL
#define SYSCALL_ERROR_LABEL 0f

PSEUDO (__libc_lseek64, lseek, 3)
	ret
PSEUDO_END (__libc_lseek64)

weak_alias (__libc_lseek64, __lseek64)
libc_hidden_weak (__lseek64)
weak_alias (__libc_lseek64, lseek64)
libc_hidden_weak (lseek64)
weak_alias (__libc_lseek64, __lseek)
libc_hidden_weak (__lseek)
weak_alias (__libc_lseek64, lseek)
