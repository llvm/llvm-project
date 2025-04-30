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
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include "posix-timer.h"


/* Create new per-process timer using CLOCK.  */
int
timer_create (clockid_t clock_id, struct sigevent *evp, timer_t *timerid)
{
  int retval = -1;
  struct timer_node *newtimer = NULL;
  struct thread_node *thread = NULL;

  if (0
#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
      || clock_id == CLOCK_PROCESS_CPUTIME_ID
#endif
#if defined _POSIX_THREAD_CPUTIME && _POSIX_THREAD_CPUTIME >= 0
      || clock_id == CLOCK_THREAD_CPUTIME_ID
#endif
      )
    {
      /* We don't allow timers for CPU clocks.  At least not in the
	 moment.  */
      __set_errno (ENOTSUP);
      return -1;
    }

  if (clock_id != CLOCK_REALTIME)
    {
      __set_errno (EINVAL);
      return -1;
    }

  pthread_once (&__timer_init_once_control, __timer_init_once);

  if (__timer_init_failed)
    {
      __set_errno (ENOMEM);
      return -1;
    }

  pthread_mutex_lock (&__timer_mutex);

  newtimer = __timer_alloc ();
  if (__glibc_unlikely (newtimer == NULL))
    {
      __set_errno (EAGAIN);
      goto unlock_bail;
    }

  if (evp != NULL)
    newtimer->event = *evp;
  else
    {
      newtimer->event.sigev_notify = SIGEV_SIGNAL;
      newtimer->event.sigev_signo = SIGALRM;
      newtimer->event.sigev_value.sival_ptr = newtimer;
      newtimer->event.sigev_notify_function = 0;
    }

  newtimer->event.sigev_notify_attributes = &newtimer->attr;
  newtimer->creator_pid = getpid ();

  switch (__builtin_expect (newtimer->event.sigev_notify, SIGEV_SIGNAL))
    {
    case SIGEV_NONE:
    case SIGEV_SIGNAL:
      /* We have a global thread for delivering timed signals.
	 If it is not running, try to start it up.  */
      thread = &__timer_signal_thread_rclk;
      if (! thread->exists)
	{
	  if (__builtin_expect (__timer_thread_start (thread),
				1) < 0)
	    {
	      __set_errno (EAGAIN);
	      goto unlock_bail;
            }
        }
      break;

    case SIGEV_THREAD:
      /* Copy over thread attributes or set up default ones.  */
      if (evp->sigev_notify_attributes)
	newtimer->attr = *(pthread_attr_t *) evp->sigev_notify_attributes;
      else
	pthread_attr_init (&newtimer->attr);

      /* Ensure thread attributes call for deatched thread.  */
      pthread_attr_setdetachstate (&newtimer->attr, PTHREAD_CREATE_DETACHED);

      /* Try to find existing thread having the right attributes.  */
      thread = __timer_thread_find_matching (&newtimer->attr, clock_id);

      /* If no existing thread has these attributes, try to allocate one.  */
      if (thread == NULL)
	thread = __timer_thread_alloc (&newtimer->attr, clock_id);

      /* Out of luck; no threads are available.  */
      if (__glibc_unlikely (thread == NULL))
	{
	  __set_errno (EAGAIN);
	  goto unlock_bail;
	}

      /* If the thread is not running already, try to start it.  */
      if (! thread->exists
	  && __builtin_expect (! __timer_thread_start (thread), 0))
	{
	  __set_errno (EAGAIN);
	  goto unlock_bail;
	}
      break;

    default:
      __set_errno (EINVAL);
      goto unlock_bail;
    }

  newtimer->clock = clock_id;
  newtimer->abstime = 0;
  newtimer->armed = 0;
  newtimer->thread = thread;

  *timerid = timer_ptr2id (newtimer);
  retval = 0;

  if (__builtin_expect (retval, 0) == -1)
    {
    unlock_bail:
      if (thread != NULL)
	__timer_thread_dealloc (thread);
      if (newtimer != NULL)
	{
	  timer_delref (newtimer);
	  __timer_dealloc (newtimer);
	}
    }

  pthread_mutex_unlock (&__timer_mutex);

  return retval;
}
