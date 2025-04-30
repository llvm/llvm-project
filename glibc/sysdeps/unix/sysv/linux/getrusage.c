/* getrusage -- get the rusage struct.  Linux version.
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
   <http://www.gnu.org/licenses/>.  */

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sysdep.h>
#include <tv32-compat.h>

int
__getrusage64 (enum __rusage_who who, struct __rusage64 *usage)
{
#if __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64
  return INLINE_SYSCALL_CALL (getrusage, who, usage);
#else
  struct __rusage32 usage32;
  if (INLINE_SYSCALL_CALL (getrusage, who, &usage32) == -1)
    return -1;

  rusage32_to_rusage64 (&usage32, usage);
  return 0;
#endif
}

#if __TIMESIZE != 64
libc_hidden_def (__getrusage64)
int
__getrusage (enum __rusage_who who, struct rusage *usage)
{
  int ret ;
  struct __rusage64 usage64;

  ret = __getrusage64 (who, &usage64);

  if (ret != 0)
    return ret;

  rusage64_to_rusage (&usage64, usage);

  return ret;
}
#endif
weak_alias (__getrusage, getrusage)
