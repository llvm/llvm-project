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
#include <limits.h>
#include <signal.h>
#include <stddef.h>	/* For NULL.  */
#include <sys/poll.h>
#include <sysdep-cancel.h>
#include <time.h>


int
ppoll (struct pollfd *fds, nfds_t nfds, const struct timespec *timeout,
       const sigset_t *sigmask)
{
  int tval = -1;

  /* poll uses a simple millisecond value.  Convert it.  */
  if (timeout != NULL)
    {
      if (timeout->tv_sec < 0 || ! valid_nanoseconds (timeout->tv_nsec))
	{
	  __set_errno (EINVAL);
	  return -1;
	}

      if (timeout->tv_sec > INT_MAX / 1000
	  || (timeout->tv_sec == INT_MAX / 1000
	      && ((timeout->tv_nsec + 999999) / 1000000 > INT_MAX % 1000)))
	/* We cannot represent the timeout in an int value.  Wait
	   forever.  */
	tval = -1;
      else
	tval = (timeout->tv_sec * 1000
		+ (timeout->tv_nsec + 999999) / 1000000);
    }

  /* The setting and restoring of the signal mask and the select call
     should be an atomic operation.  This can't be done without kernel
     help.  */
  sigset_t savemask;
  if (sigmask != NULL)
    __sigprocmask (SIG_SETMASK, sigmask, &savemask);

  /* Note the ppoll() is a cancellation point.  But since we call
     poll() which itself is a cancellation point we do not have
     to do anything here.  */
  int retval = __poll (fds, nfds, tval);

  if (sigmask != NULL)
    __sigprocmask (SIG_SETMASK, &savemask, NULL);

  return retval;
}

#ifndef ppoll
libc_hidden_def (ppoll);
#endif
