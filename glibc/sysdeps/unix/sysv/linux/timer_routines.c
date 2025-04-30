/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <sysdep-cancel.h>
#include <pthreadP.h>
#include "kernel-posix-timers.h"


/* List of active SIGEV_THREAD timers.  */
struct timer *__timer_active_sigev_thread;

/* Lock for _timer_active_sigev_thread.  */
pthread_mutex_t __timer_active_sigev_thread_lock = PTHREAD_MUTEX_INITIALIZER;

struct thread_start_data
{
  void (*thrfunc) (sigval_t);
  sigval_t sival;
};


/* Helper thread to call the user-provided function.  */
static void *
timer_sigev_thread (void *arg)
{
  __libc_signal_unblock_sigtimer (NULL);

  struct thread_start_data *td = (struct thread_start_data *) arg;
  void (*thrfunc) (sigval_t) = td->thrfunc;
  sigval_t sival = td->sival;

  /* The TD object was allocated in timer_helper_thread.  */
  free (td);

  /* Call the user-provided function.  */
  thrfunc (sival);

  return NULL;
}


/* Helper function to support starting threads for SIGEV_THREAD.  */
static _Noreturn void *
timer_helper_thread (void *arg)
{
  /* Endless loop of waiting for signals.  The loop is only ended when
     the thread is canceled.  */
  while (1)
    {
      siginfo_t si;

      while (__sigwaitinfo (&sigtimer_set, &si) < 0);
      if (si.si_code == SI_TIMER)
	{
	  struct timer *tk = (struct timer *) si.si_ptr;

	  /* Check the timer is still used and will not go away
	     while we are reading the values here.  */
	  __pthread_mutex_lock (&__timer_active_sigev_thread_lock);

	  struct timer *runp = __timer_active_sigev_thread;
	  while (runp != NULL)
	    if (runp == tk)
	      break;
	  else
	    runp = runp->next;

	  if (runp != NULL)
	    {
	      struct thread_start_data *td = malloc (sizeof (*td));

	      /* There is not much we can do if the allocation fails.  */
	      if (td != NULL)
		{
		  /* This is the signal we are waiting for.  */
		  td->thrfunc = tk->thrfunc;
		  td->sival = tk->sival;

		  pthread_t th;
		  __pthread_create (&th, &tk->attr, timer_sigev_thread, td);
		}
	    }

	  __pthread_mutex_unlock (&__timer_active_sigev_thread_lock);
	}
    }
}


/* Control variable for helper thread creation.  */
pthread_once_t __timer_helper_once = PTHREAD_ONCE_INIT;


/* TID of the helper thread.  */
pid_t __timer_helper_tid;


/* Reset variables so that after a fork a new helper thread gets started.  */
void
__timer_fork_subprocess (void)
{
  __timer_helper_once = PTHREAD_ONCE_INIT;
  __timer_helper_tid = 0;
}


void
__timer_start_helper_thread (void)
{
  /* The helper thread needs only very little resources
     and should go away automatically when canceled.  */
  pthread_attr_t attr;
  __pthread_attr_init (&attr);
  __pthread_attr_setstacksize (&attr, __pthread_get_minstack (&attr));

  /* Block all signals in the helper thread but SIGSETXID.  */
  sigset_t ss;
  __sigfillset (&ss);
  __sigdelset (&ss, SIGSETXID);
  int res = __pthread_attr_setsigmask_internal (&attr, &ss);
  if (res != 0)
    {
      __pthread_attr_destroy (&attr);
      return;
    }

  /* Create the helper thread for this timer.  */
  pthread_t th;
  res = __pthread_create (&th, &attr, timer_helper_thread, NULL);
  if (res == 0)
    /* We managed to start the helper thread.  */
    __timer_helper_tid = ((struct pthread *) th)->tid;

  /* No need for the attribute anymore.  */
  __pthread_attr_destroy (&attr);
}
