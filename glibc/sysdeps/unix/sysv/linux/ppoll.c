/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2006.

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
#include <signal.h>
#include <time.h>
#include <sys/poll.h>
#include <sysdep-cancel.h>

int
__ppoll64 (struct pollfd *fds, nfds_t nfds, const struct __timespec64 *timeout,
           const sigset_t *sigmask)
{
  /* The Linux kernel can in some situations update the timeout value.
     We do not want that so use a local variable.  */
  struct __timespec64 tval;
  if (timeout != NULL)
    {
      tval = *timeout;
      timeout = &tval;
    }

#ifndef __NR_ppoll_time64
# define __NR_ppoll_time64 __NR_ppoll
#endif

#ifdef __ASSUME_TIME64_SYSCALLS
  return SYSCALL_CANCEL (ppoll_time64, fds, nfds, timeout, sigmask,
			 __NSIG_BYTES);
#else
  int ret;
  bool need_time64 = timeout != NULL && !in_time_t_range (timeout->tv_sec);
  if (need_time64)
    {
      ret = SYSCALL_CANCEL (ppoll_time64, fds, nfds, timeout, sigmask,
			    __NSIG_BYTES);
      if (ret == 0 || errno != ENOSYS)
	return ret;
      __set_errno (EOVERFLOW);
      return -1;
    }

  struct timespec ts32;
  if (timeout != NULL)
    ts32 = valid_timespec64_to_timespec (*timeout);

  return SYSCALL_CANCEL (ppoll, fds, nfds, timeout ? &ts32 : NULL, sigmask,
			 __NSIG_BYTES);
#endif
}

#if __TIMESIZE != 64
libc_hidden_def (__ppoll64)

int
__ppoll (struct pollfd *fds, nfds_t nfds, const struct timespec *timeout,
         const sigset_t *sigmask)
{
  struct __timespec64 ts64;
  if (timeout)
    ts64 = valid_timespec_to_timespec64 (*timeout);

  return __ppoll64 (fds, nfds, timeout ? &ts64 : NULL, sigmask);
}
#endif
strong_alias (__ppoll, ppoll)
libc_hidden_def (ppoll)
