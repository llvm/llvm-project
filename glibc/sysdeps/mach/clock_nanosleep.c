/* clock_nanosleep - high-resolution sleep with specifiable clock.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <mach.h>
#include <time.h>
#include <unistd.h>
#include <posix-timer.h>
#include <shlib-compat.h>
#include <sysdep-cancel.h>

static int
nanosleep_call (const struct timespec *req, struct timespec *rem)
{
  mach_port_t recv;
  struct timespec before;
  error_t err;

  const mach_msg_timeout_t ms
    = req->tv_sec * 1000
    + (req->tv_nsec + 999999) / 1000000;

  recv = __mach_reply_port ();

  if (rem != NULL)
    __clock_gettime (CLOCK_REALTIME, &before);

  int cancel_oldtype = LIBC_CANCEL_ASYNC();
  err = __mach_msg (NULL, MACH_RCV_MSG|MACH_RCV_TIMEOUT|MACH_RCV_INTERRUPT,
                    0, 0, recv, ms, MACH_PORT_NULL);
  LIBC_CANCEL_RESET (cancel_oldtype);

  __mach_port_destroy (mach_task_self (), recv);

  if (err == EMACH_RCV_INTERRUPTED)
    {
      if (rem != NULL)
	{
	  struct timespec after, elapsed;
	  __clock_gettime (CLOCK_REALTIME, &after);
	  timespec_sub (&elapsed, &after, &before);
	  timespec_sub (rem, req, &elapsed);
	}

      return EINTR;
    }

  return 0;
}

int
__clock_nanosleep (clockid_t clock_id, int flags, const struct timespec *req,
		   struct timespec *rem)
{
  if (clock_id != CLOCK_REALTIME
      || !valid_nanoseconds (req->tv_nsec)
      || (flags != 0 && flags != TIMER_ABSTIME))
    return EINVAL;

  struct timespec now;

  /* If we got an absolute time, remap it.  */
  if (flags == TIMER_ABSTIME)
    {
      long int nsec;
      long int sec;

      /* Make sure we use safe data types.  */
      assert (sizeof (sec) >= sizeof (now.tv_sec));

      /* Get the current time for this clock.  */
      if (__clock_gettime (clock_id, &now) != 0)
	return errno;

      /* Compute the difference.  */
      nsec = req->tv_nsec - now.tv_nsec;
      sec = req->tv_sec - now.tv_sec - (nsec < 0);
      if (sec < 0)
	/* The time has already elapsed.  */
	return 0;

      now.tv_sec = sec;
      now.tv_nsec = nsec + (nsec < 0 ? 1000000000 : 0);

      /* From now on this is our time.  */
      req = &now;

      /* Make sure we are not modifying the struct pointed to by REM.  */
      rem = NULL;
    }

  return nanosleep_call (req, rem);
}
libc_hidden_def (__clock_nanosleep)
versioned_symbol (libc, __clock_nanosleep, clock_nanosleep, GLIBC_2_17);
/* clock_nanosleep moved to libc in version 2.17;
   old binaries may expect the symbol version it had in librt.  */
#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
strong_alias (__clock_nanosleep, __clock_nanosleep_2);
compat_symbol (libc, __clock_nanosleep_2, clock_nanosleep, GLIBC_2_2);
#endif
