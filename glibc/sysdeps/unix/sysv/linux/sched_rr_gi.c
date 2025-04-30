/* sched_rr_get_interval -- get the scheduler's SCHED_RR policy time interval.
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
__sched_rr_get_interval64 (pid_t pid, struct __timespec64 *tp)
{
#ifndef __NR_sched_rr_get_interval_time64
# define __NR_sched_rr_get_interval_time64 __NR_sched_rr_get_interval
#endif
  int ret = INLINE_SYSCALL_CALL (sched_rr_get_interval_time64, pid, tp);
#ifndef __ASSUME_TIME64_SYSCALLS
  if (ret == 0 || errno != ENOSYS)
    return ret;

  struct timespec tp32;
  ret = INLINE_SYSCALL_CALL (sched_rr_get_interval, pid, &tp32);
  if (ret == 0)
    *tp = valid_timespec_to_timespec64 (tp32);
#endif
  return ret;
}

#if __TIMESIZE != 64
libc_hidden_def (__sched_rr_get_interval64)

int
__sched_rr_get_interval (pid_t pid, struct timespec *tp)
{
  int ret;
  struct __timespec64 tp64;

  ret = __sched_rr_get_interval64 (pid, &tp64);

  if (ret == 0)
    {
      if (! in_time_t_range (tp64.tv_sec))
        {
          __set_errno (EOVERFLOW);
          return -1;
        }

      *tp = valid_timespec64_to_timespec (tp64);
    }

  return ret;
}
#endif
strong_alias (__sched_rr_get_interval, sched_rr_get_interval)
