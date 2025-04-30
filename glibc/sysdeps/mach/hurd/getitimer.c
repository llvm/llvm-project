/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <stddef.h>
#include <errno.h>
#include <sys/time.h>
#include <hurd.h>
#include <mach.h>

/* XXX Temporary cheezoid implementation; see setitimer.c.  */

/* These are defined in __setitmr.c.  */
extern spin_lock_t _hurd_itimer_lock;
extern struct itimerval _hurd_itimerval;
extern struct timeval _hurd_itimer_started;

static inline void
subtract_timeval (struct timeval *from, const struct timeval *subtract)
{
  from->tv_usec -= subtract->tv_usec;
  from->tv_sec -= subtract->tv_sec;
  while (from->tv_usec < 0)
    {
      --from->tv_sec;
      from->tv_usec += 1000000;
    }
}

/* Set *VALUE to the current setting of timer WHICH.
   Return 0 on success, -1 on errors.  */
int
__getitimer (enum __itimer_which which, struct itimerval *value)
{
  struct itimerval val;
  struct timeval elapsed;

  switch (which)
    {
    default:
      return __hurd_fail (EINVAL);

    case ITIMER_VIRTUAL:
    case ITIMER_PROF:
      return __hurd_fail (ENOSYS);

    case ITIMER_REAL:
      break;
    }

  /* Get the time now.  */
  {
     time_value_t tv;
     __host_get_time (__mach_host_self (), &tv);
     elapsed.tv_sec = tv.seconds;
     elapsed.tv_usec = tv.microseconds;
  }

  /* Extract the current timer setting; and the time it was set, so we can
     calculate the time elapsed so far.  */
  HURD_CRITICAL_BEGIN;
  __spin_lock (&_hurd_itimer_lock);
  val = _hurd_itimerval;
  subtract_timeval (&elapsed, &_hurd_itimer_started);
  __spin_unlock (&_hurd_itimer_lock);
  HURD_CRITICAL_END;

  if ((val.it_value.tv_sec | val.it_value.tv_usec) != 0)
    {
      /* There is a pending alarm set.  VAL indicates the interval it was
	 set for, relative to the time recorded in _hurd_itimer_started.
	 Now compensate for the time elapsed since to get the user's
	 conception of the current value of the timer (as if the value
	 stored decreased every microsecond).  */
      if (timercmp (&val.it_value, &elapsed, <))
	{
	  /* Hmm.  The timer should have just gone off, but has not been
	     reset.  This is a possible timing glitch.  The alarm will signal
	     soon, so fabricate a value for how soon.  */
	  val.it_value.tv_sec = 0;
	  val.it_value.tv_usec = 10; /* Random.  */
	}
      else
	/* Subtract the time elapsed since the timer was set
	   from the current timer value the user sees.  */
	subtract_timeval (&val.it_value, &elapsed);
    }

  *value = val;
  return 0;
}

weak_alias (__getitimer, getitimer)
