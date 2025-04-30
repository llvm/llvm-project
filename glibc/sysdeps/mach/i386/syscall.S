/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

ENTRY (syscall)
	popl %ecx		/* Pop return address into %ecx.  */
	popl %eax		/* Pop syscall number into %eax.  */
	pushl %ecx		/* Push back return address.  */
	.byte 0x9a, 0, 0, 0, 0, 7, 0 /* lcall $7, $0 -- gas bug */
	popl %ecx		/* Pop return address into %ecx.  */
	pushl $0		/* Push back dumb syscall number.  */
	pushl %ecx		/* Push back return address.  */
	ret
END (syscall)
