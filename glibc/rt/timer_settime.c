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


/* Set timer TIMERID to VALUE, returning old value in OVLAUE.  */
int
timer_settime (timer_t timerid, int flags, const struct itimerspec *value,
	       struct itimerspec *ovalue)
{
  struct timer_node *timer;
  struct thread_node *thread = NULL;
  struct timespec now;
  int have_now = 0, need_wakeup = 0;
  int retval = -1;

  timer = timer_id2ptr (timerid);
  if (timer == NULL)
    {
      __set_errno (EINVAL);
      goto bail;
    }

  if (! valid_nanoseconds (value->it_interval.tv_nsec)
      || ! valid_nanoseconds (value->it_value.tv_nsec))
    {
      __set_errno (EINVAL);
      goto bail;
    }

  /* Will need to know current time since this is a relative timer;
     might as well make the system call outside of the lock now! */

  if ((flags & TIMER_ABSTIME) == 0)
    {
      __clock_gettime (timer->clock, &now);
      have_now = 1;
    }

  pthread_mutex_lock (&__timer_mutex);
  timer_addref (timer);

  /* One final check of timer validity; this one is possible only
     until we have the mutex, because it accesses the inuse flag. */

  if (! timer_valid(timer))
    {
      __set_errno (EINVAL);
      goto unlock_bail;
    }

  if (ovalue != NULL)
    {
      ovalue->it_interval = timer->value.it_interval;

      if (timer->armed)
	{
	  if (! have_now)
	    {
	      pthread_mutex_unlock (&__timer_mutex);
	      __clock_gettime (timer->clock, &now);
	      have_now = 1;
	      pthread_mutex_lock (&__timer_mutex);
	      timer_addref (timer);
	    }

	  timespec_sub (&ovalue->it_value, &timer->expirytime, &now);
	}
      else
	{
	  ovalue->it_value.tv_sec = 0;
	  ovalue->it_value.tv_nsec = 0;
	}
    }

  timer->value = *value;

  list_unlink_ip (&timer->links);
  timer->armed = 0;

  thread = timer->thread;

  /* A value of { 0, 0 } causes the timer to be stopped. */
  if (value->it_value.tv_sec != 0
      || __builtin_expect (value->it_value.tv_nsec != 0, 1))
    {
      if ((flags & TIMER_ABSTIME) != 0)
	/* The user specified the expiration time.  */
	timer->expirytime = value->it_value;
      else
	timespec_add (&timer->expirytime, &now, &value->it_value);

      /* Only need to wake up the thread if timer is inserted
	 at the head of the queue. */
      if (thread != NULL)
	need_wakeup = __timer_thread_queue_timer (thread, timer);
      timer->armed = 1;
    }

  retval = 0;

unlock_bail:
  timer_delref (timer);
  pthread_mutex_unlock (&__timer_mutex);

bail:
  if (thread != NULL && need_wakeup)
    __timer_thread_wakeup (thread);

  return retval;
}
