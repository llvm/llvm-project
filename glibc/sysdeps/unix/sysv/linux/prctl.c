/* prctl - Linux specific syscall.
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

#include <sysdep.h>
#include <stdarg.h>
#include <sys/prctl.h>

/* Unconditionally read all potential arguments.  This may pass
   garbage values to the kernel, but avoids the need for teaching
   glibc the argument counts of individual options (including ones
   that are added to the kernel in the future).  */

int
__prctl (int option, ...)
{
  va_list arg;
  va_start (arg, option);
  unsigned long int arg2 = va_arg (arg, unsigned long int);
  unsigned long int arg3 = va_arg (arg, unsigned long int);
  unsigned long int arg4 = va_arg (arg, unsigned long int);
  unsigned long int arg5 = va_arg (arg, unsigned long int);
  va_end (arg);
  return INLINE_SYSCALL_CALL (prctl, option, arg2, arg3, arg4, arg5);
}

libc_hidden_def (__prctl)
weak_alias (__prctl, prctl)
#if __TIMESIZE != 64
weak_alias (__prctl, __prctl_time64)
#endif
