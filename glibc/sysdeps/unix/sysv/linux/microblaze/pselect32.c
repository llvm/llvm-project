/* Synchronous I/O multiplexing.  Linux/microblaze version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
#include <signal.h>
#include <time.h>
#include <sys/poll.h>
#include <sysdep-cancel.h>

#ifndef __ASSUME_PSELECT
int
__pselect32 (int nfds, fd_set *readfds, fd_set *writefds,
	     fd_set *exceptfds, const struct __timespec64 *timeout,
	     const sigset_t *sigmask)
{
  /* The fallback uses 'select' which shows the race condition regarding
     signal mask set/restore, requires two additional syscalls, and has
     a worse timeout precision (microseconds instead of nanoseconds).  */

  struct timeval tv32, *ptv32 = NULL;
  if (timeout != NULL)
    {
      if (! valid_nanoseconds (timeout->tv_nsec))
	{
	  __set_errno (EINVAL);
	  return -1;
	}

      tv32 = valid_timespec64_to_timeval (*timeout);
      ptv32 = &tv32;
    }

  sigset_t savemask;
  if (sigmask != NULL)
    __sigprocmask (SIG_SETMASK, sigmask, &savemask);

  /* select itself is a cancellation entrypoint.  */
  int ret = __select (nfds, readfds, writefds, exceptfds, ptv32);

  if (sigmask != NULL)
    __sigprocmask (SIG_SETMASK, &savemask, NULL);

  return ret;
}
#endif /* __ASSUME_PSELECT  */
