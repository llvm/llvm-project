/* MIPS16 syscall wrappers.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#undef __mips16_syscall4

long long int __nomips16
__mips16_syscall4 (long int a0, long int a1, long int a2, long int a3,
		   long int number)
{
  union __mips_syscall_return ret;
  ret.reg.v0 = INTERNAL_SYSCALL_MIPS16 (number, ret.reg.v1, 4,
					a0, a1, a2, a3);
  return ret.val;
}
