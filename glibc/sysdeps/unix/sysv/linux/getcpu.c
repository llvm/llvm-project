/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <sched.h>
#include <sysdep.h>
#include <sysdep-vdso.h>

int
__getcpu (unsigned int *cpu, unsigned int *node)
{
#ifdef HAVE_GETCPU_VSYSCALL
  return INLINE_VSYSCALL (getcpu, 3, cpu, node, NULL);
#else
  return INLINE_SYSCALL_CALL (getcpu, cpu, node, NULL);
#endif
}
weak_alias (__getcpu, getcpu)
libc_hidden_def (__getcpu)
