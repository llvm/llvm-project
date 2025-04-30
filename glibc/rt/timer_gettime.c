/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kaz Kylheku <kaz@ashi.footprints.net>.

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
#include <pthread.h>
#include <time.h>

#include "posix-timer.h"


/* Get current value of timer TIMERID and store it in VLAUE.  */
int
timer_gettime (timer_t timerid, struct itimerspec *value)
{
  struct timer_node *timer;
  struct timespec now, expiry;
  int retval = -1, armed = 0, valid;
  clock_t clock = 0;

  pthread_mutex_lock (&__timer_mutex);

  timer = timer_id2ptr (timerid);
  valid = timer_valid (timer);

  if (valid) {
    armed = timer->armed;
    expiry = timer->expirytime;
    clock = timer->clock;
    value->it_interval = timer->value.it_interval;
  }

  pthread_mutex_unlock (&__timer_mutex);

  if (valid)
    {
      if (armed)
	{
	  __clock_gettime (clock, &now);
	  if (timespec_compare (&now, &expiry) < 0)
	    timespec_sub (&value->it_value, &expiry, &now);
	  else
	    {
	      value->it_value.tv_sec = 0;
	      value->it_value.tv_nsec = 0;
	    }
	}
      else
	{
	  value->it_value.tv_sec = 0;
	  value->it_value.tv_nsec = 0;
	}

      retval = 0;
    }
  else
    __set_errno (EINVAL);

  return retval;
}
