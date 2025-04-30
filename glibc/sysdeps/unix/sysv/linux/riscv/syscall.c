/* system call interface.  Linux/RISC-V version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

long int
syscall (long int syscall_number, long int arg1, long int arg2, long int arg3,
	 long int arg4, long int arg5, long int arg6, long int arg7)
{
  long int ret;

  ret = INTERNAL_SYSCALL_NCS (syscall_number, 7, arg1, arg2, arg3, arg4,
			      arg5, arg6, arg7);

  if (INTERNAL_SYSCALL_ERROR_P (ret))
    return __syscall_error (ret);

  return ret;
}
