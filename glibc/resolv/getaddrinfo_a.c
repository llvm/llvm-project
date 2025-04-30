/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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
#include <netdb.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#include <gai_misc.h>


/* We need this special structure to handle asynchronous I/O.  */
struct async_waitlist
  {
    unsigned int counter;
    struct sigevent sigev;
    struct waitlist list[0];
  };


int
__getaddrinfo_a (int mode, struct gaicb *list[], int ent, struct sigevent *sig)
{
  struct sigevent defsigev;
  struct requestlist *requests[ent];
  int cnt;
  volatile unsigned int total = 0;
  int result = 0;

  /* Check arguments.  */
  if (mode != GAI_WAIT && mode != GAI_NOWAIT)
    {
      __set_errno (EINVAL);
      return EAI_SYSTEM;
    }

  if (sig == NULL)
    {
      defsigev.sigev_notify = SIGEV_NONE;
      sig = &defsigev;
    }

  /* Request the mutex.  */
  __pthread_mutex_lock (&__gai_requests_mutex);

  /* Now we can enqueue all requests.  Since we already acquired the
     mutex the enqueue function need not do this.  */
  for (cnt = 0; cnt < ent; ++cnt)
    if (list[cnt] != NULL)
      {
	requests[cnt] = __gai_enqueue_request (list[cnt]);

	if (requests[cnt] != NULL)
	  /* Successfully enqueued.  */
	  ++total;
	else
	  /* Signal that we've seen an error.  `errno' and the error code
	     of the gaicb will tell more.  */
	  result = EAI_SYSTEM;
      }
    else
      requests[cnt] = NULL;

  if (total == 0)
    {
      /* We don't have anything to do except signalling if we work
	 asynchronously.  */

      /* Release the mutex.  We do this before raising a signal since the
	 signal handler might do a `siglongjmp' and then the mutex is
	 locked forever.  */
      __pthread_mutex_unlock (&__gai_requests_mutex);

      if (mode == GAI_NOWAIT)
	__gai_notify_only (sig,
			   sig->sigev_notify == SIGEV_SIGNAL ? getpid () : 0);

      return result;
    }
  else if (mode == GAI_WAIT)
    {
#ifndef DONT_NEED_GAI_MISC_COND
      pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
#endif
      struct waitlist waitlist[ent];
      int oldstate;

      total = 0;
      for (cnt = 0; cnt < ent; ++cnt)
	if (requests[cnt] != NULL)
	  {
#ifndef DONT_NEED_GAI_MISC_COND
	    waitlist[cnt].cond = &cond;
#endif
	    waitlist[cnt].next = requests[cnt]->waiting;
	    waitlist[cnt].counterp = &total;
	    waitlist[cnt].sigevp = NULL;
	    waitlist[cnt].caller_pid = 0;	/* Not needed.  */
	    requests[cnt]->waiting = &waitlist[cnt];
	    ++total;
	  }

      /* Since `pthread_cond_wait'/`pthread_cond_timedwait' are cancelation
	 points we must be careful.  We added entries to the waiting lists
	 which we must remove.  So defer cancelation for now.  */
      __pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &oldstate);

      while (total > 0)
	{
#ifdef DONT_NEED_GAI_MISC_COND
	  int not_used __attribute__ ((unused));
	  GAI_MISC_WAIT (not_used, total, NULL, 1);
#else
	  pthread_cond_wait (&cond, &__gai_requests_mutex);
#endif
	}

      /* Now it's time to restore the cancelation state.  */
      __pthread_setcancelstate (oldstate, NULL);

#ifndef DONT_NEED_GAI_MISC_COND
      /* Release the conditional variable.  */
      if (pthread_cond_destroy (&cond) != 0)
	/* This must never happen.  */
	abort ();
#endif
    }
  else
    {
      struct async_waitlist *waitlist;

      waitlist = (struct async_waitlist *)
	malloc (sizeof (struct async_waitlist)
		+ (ent * sizeof (struct waitlist)));

      if (waitlist == NULL)
	result = EAI_AGAIN;
      else
	{
	  pid_t caller_pid = sig->sigev_notify == SIGEV_SIGNAL ? getpid () : 0;
	  total = 0;

	  for (cnt = 0; cnt < ent; ++cnt)
	    if (requests[cnt] != NULL)
	      {
#ifndef DONT_NEED_GAI_MISC_COND
		waitlist->list[cnt].cond = NULL;
#endif
		waitlist->list[cnt].next = requests[cnt]->waiting;
		waitlist->list[cnt].counterp = &waitlist->counter;
		waitlist->list[cnt].sigevp = &waitlist->sigev;
		waitlist->list[cnt].caller_pid = caller_pid;
		requests[cnt]->waiting = &waitlist->list[cnt];
		++total;
	      }

	  waitlist->counter = total;
	  waitlist->sigev = *sig;
	}
    }

  /* Release the mutex.  */
  __pthread_mutex_unlock (&__gai_requests_mutex);

  return result;
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __getaddrinfo_a, getaddrinfo_a, GLIBC_2_34);

# if OTHER_SHLIB_COMPAT (libanl, GLIBC_2_2_3, GLIBC_2_34)
compat_symbol (libanl, __getaddrinfo_a, getaddrinfo_a, GLIBC_2_2_3);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__getaddrinfo_a, getaddrinfo_a)
#endif /* !PTHREAD_IN_LIBC */
