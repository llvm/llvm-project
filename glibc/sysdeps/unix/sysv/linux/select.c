/* Linux select implementation.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <sys/time.h>
#include <sys/types.h>
#include <sys/select.h>
#include <errno.h>
#include <sysdep-cancel.h>

/* Check the first NFDS descriptors each in READFDS (if not NULL) for read
   readiness, in WRITEFDS (if not NULL) for write readiness, and in EXCEPTFDS
   (if not NULL) for exceptional conditions.  If TIMEOUT is not NULL, time out
   after waiting the interval specified therein.  Returns the number of ready
   descriptors, or -1 for errors.  */

int
__select64 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
	    struct __timeval64 *timeout)
{
  __time64_t s = timeout != NULL ? timeout->tv_sec : 0;
  int32_t us = timeout != NULL ? timeout->tv_usec : 0;
  int32_t ns;

  if (s < 0 || us < 0)
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

  /* Normalize the timeout, as legacy Linux __NR_select and __NR__newselect.
     Different than syscall, it also handle possible overflow.  */
  if (us / USEC_PER_SEC > INT64_MAX - s)
    {
      s = INT64_MAX;
      ns = NSEC_PER_SEC - 1;
    }
  else
    {
      s += us / USEC_PER_SEC;
      us = us % USEC_PER_SEC;
      ns = us * NSEC_PER_USEC;
    }

  struct __timespec64 ts64, *pts64 = NULL;
   if (timeout != NULL)
     {
       ts64.tv_sec = s;
       ts64.tv_nsec = ns;
       pts64 = &ts64;
     }

#ifndef __NR_pselect6_time64
# define __NR_pselect6_time64 __NR_pselect6
#endif

#ifdef __ASSUME_TIME64_SYSCALLS
  int r = SYSCALL_CANCEL (pselect6_time64, nfds, readfds, writefds, exceptfds,
			  pts64, NULL);
  if (timeout != NULL)
    TIMESPEC_TO_TIMEVAL (timeout, pts64);
  return r;
#else
  bool need_time64 = timeout != NULL && !in_time_t_range (timeout->tv_sec);
  if (need_time64)
    {
      int r = SYSCALL_CANCEL (pselect6_time64, nfds, readfds, writefds,
			      exceptfds, pts64, NULL);
      if ((r >= 0 || errno != ENOSYS) && timeout != NULL)
	{
	  TIMESPEC_TO_TIMEVAL (timeout, &ts64);
	}
      else
	__set_errno (EOVERFLOW);
      return r;
    }

# ifdef __ASSUME_PSELECT
  struct timespec ts32, *pts32 = NULL;
  if (pts64 != NULL)
    {
      ts32.tv_sec = pts64->tv_sec;
      ts32.tv_nsec = pts64->tv_nsec;
      pts32 = &ts32;
    }

  int r = SYSCALL_CANCEL (pselect6, nfds, readfds, writefds, exceptfds, pts32,
			  NULL);
  if (timeout != NULL)
    TIMESPEC_TO_TIMEVAL (timeout, pts32);
  return r;
# else
  struct timeval tv32, *ptv32 = NULL;
  if (pts64 != NULL)
    {
      tv32 = valid_timespec64_to_timeval (*pts64);
      ptv32 = &tv32;
    }

  int r = SYSCALL_CANCEL (select, nfds, readfds, writefds, exceptfds, ptv32);
  if (timeout != NULL)
    *timeout = valid_timeval_to_timeval64 (tv32);
  return r;
# endif /* __ASSUME_PSELECT  */
#endif
}

#if __TIMESIZE != 64
libc_hidden_def (__select64)

int
__select (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
	  struct timeval *timeout)
{
  struct __timeval64 tv64, *ptv64 = NULL;
  if (timeout != NULL)
    {
      tv64 = valid_timeval_to_timeval64 (*timeout);
      ptv64 = &tv64;
    }
  int r = __select64 (nfds, readfds, writefds, exceptfds, ptv64);
  if (timeout != NULL)
    /* The remanining timeout will be always less the input TIMEOUT.  */
    *timeout = valid_timeval64_to_timeval (tv64);
  return r;
}
#endif
libc_hidden_def (__select)

weak_alias (__select, select)
weak_alias (__select, __libc_select)
