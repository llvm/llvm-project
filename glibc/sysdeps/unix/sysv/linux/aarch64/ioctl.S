/* Copyright (C) 2012-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.text
ENTRY(__ioctl)
	mov	x8, #__NR_ioctl
	sxtw	x0, w0
	svc	#0x0
	cmn	x0, #4095
	b.cs	.Lsyscall_error
	ret
PSEUDO_END (__ioctl)

libc_hidden_def (__ioctl)
weak_alias (__ioctl, ioctl)
