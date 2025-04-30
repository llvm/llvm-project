/* timerfd_gettime -- get the timer setting referred to by file descriptor.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <sysdep.h>
#include <kernel-features.h>

int
__timerfd_gettime64 (int fd, struct __itimerspec64 *value)
{
#ifndef __NR_timerfd_gettime64
# define __NR_timerfd_gettime64 __NR_timerfd_gettime
#endif

#ifdef __ASSUME_TIME64_SYSCALLS
  return INLINE_SYSCALL_CALL (timerfd_gettime64, fd, value);
#else
  int ret = INLINE_SYSCALL_CALL (timerfd_gettime64, fd, value);
  if (ret == 0 || errno != ENOSYS)
    return ret;
  struct itimerspec its32;
  int retval = INLINE_SYSCALL_CALL (timerfd_gettime, fd, &its32);
  if (retval == 0)
    {
      value->it_interval = valid_timespec_to_timespec64 (its32.it_interval);
      value->it_value = valid_timespec_to_timespec64 (its32.it_value);
    }

  return retval;
#endif
}

#if __TIMESIZE != 64
libc_hidden_def (__timerfd_gettime64)

int
__timerfd_gettime (int fd, struct itimerspec *value)
{
  struct __itimerspec64 its64;
  int retval = __timerfd_gettime64 (fd, &its64);
  if (retval == 0)
    {
      value->it_interval = valid_timespec64_to_timespec (its64.it_interval);
      value->it_value = valid_timespec64_to_timespec (its64.it_value);
    }

  return retval;
}
#endif
strong_alias (__timerfd_gettime, timerfd_gettime)
