/* timerfd_settime -- start or stop the timer referred to by file descriptor.
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
__timerfd_settime64 (int fd, int flags, const struct __itimerspec64 *value,
                     struct __itimerspec64 *ovalue)
{
#ifndef __NR_timerfd_settime64
# define __NR_timerfd_settime64 __NR_timerfd_settime
#endif

#ifdef __ASSUME_TIME64_SYSCALLS
  return INLINE_SYSCALL_CALL (timerfd_settime64, fd, flags, value, ovalue);
#else
  bool need_time64 = !in_time_t_range (value->it_value.tv_sec)
		     || !in_time_t_range (value->it_interval.tv_sec);
  if (need_time64)
    {
      int r = INLINE_SYSCALL_CALL (timerfd_settime64, fd, flags, value,
				   ovalue);
      if (r == 0 || errno != ENOSYS)
	return r;
      __set_errno (EOVERFLOW);
      return r;
    }

  struct itimerspec its32, oits32;
  its32.it_interval = valid_timespec64_to_timespec (value->it_interval);
  its32.it_value = valid_timespec64_to_timespec (value->it_value);
  int ret = INLINE_SYSCALL_CALL (timerfd_settime, fd, flags,
				 &its32, ovalue != NULL ? &oits32 : NULL);
  if (ret == 0 && ovalue != NULL)
    {
      ovalue->it_interval = valid_timespec_to_timespec64 (oits32.it_interval);
      ovalue->it_value = valid_timespec_to_timespec64 (oits32.it_value);
    }
  return ret;
#endif
}

#if __TIMESIZE != 64
libc_hidden_def (__timerfd_settime64)

int
__timerfd_settime (int fd, int flags, const struct itimerspec *value,
                   struct itimerspec *ovalue)
{
  struct __itimerspec64 its64, oits64;
  int retval;

  its64.it_interval = valid_timespec_to_timespec64 (value->it_interval);
  its64.it_value = valid_timespec_to_timespec64 (value->it_value);

  retval = __timerfd_settime64 (fd, flags, &its64, ovalue ? &oits64 : NULL);
  if (retval == 0 && ovalue)
    {
      ovalue->it_interval = valid_timespec64_to_timespec (oits64.it_interval);
      ovalue->it_value = valid_timespec64_to_timespec (oits64.it_value);
    }

  return retval;
}
#endif
strong_alias (__timerfd_settime, timerfd_settime)
