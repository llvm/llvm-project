/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <signal.h>
#include <sysdep.h>

int
__sigtimedwait64 (const sigset_t *set, siginfo_t *info,
		  const struct __timespec64 *timeout)
{
#ifndef __NR_rt_sigtimedwait_time64
# define __NR_rt_sigtimedwait_time64 __NR_rt_sigtimedwait
#endif

  int result;
#ifdef __ASSUME_TIME64_SYSCALLS
  result = SYSCALL_CANCEL (rt_sigtimedwait_time64, set, info, timeout,
			   __NSIG_BYTES);
#else
  bool need_time64 = timeout != NULL && !in_time_t_range (timeout->tv_sec);
  if (need_time64)
    {
      result = SYSCALL_CANCEL (rt_sigtimedwait_time64, set, info, timeout,
			       __NSIG_BYTES);
      if (result == 0 || errno != ENOSYS)
	return result;
      __set_errno (EOVERFLOW);
      return -1;
    }
  else
    {
      struct timespec ts32, *pts32 = NULL;
      if (timeout != NULL)
	{
	  ts32 = valid_timespec64_to_timespec (*timeout);
	  pts32 = &ts32;
	}
      result = SYSCALL_CANCEL (rt_sigtimedwait, set, info, pts32,
			       __NSIG_BYTES);
    }
#endif

  /* The kernel generates a SI_TKILL code in si_code in case tkill is
     used.  tkill is transparently used in raise().  Since having
     SI_TKILL as a code is useful in general we fold the results
     here.  */
  if (result != -1 && info != NULL && info->si_code == SI_TKILL)
    info->si_code = SI_USER;

  return result;
}
#if __TIMESIZE != 64
libc_hidden_def (__sigtimedwait64)

int
__sigtimedwait (const sigset_t *set, siginfo_t *info,
		const struct timespec *timeout)
{
  struct __timespec64 ts64, *pts64 = NULL;
  if (timeout != NULL)
    {
      ts64 = valid_timespec_to_timespec64 (*timeout);
      pts64 = &ts64;
    }
  return __sigtimedwait64 (set, info, pts64);
}
#endif
libc_hidden_def (__sigtimedwait)
weak_alias (__sigtimedwait, sigtimedwait)
