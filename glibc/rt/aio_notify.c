/* Notify initiator of AIO request.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <pthreadP.h>
#include <stdlib.h>
#include <unistd.h>
#include <aio_misc.h>
#include <signal.h>

#if !PTHREAD_IN_LIBC
# define __pthread_attr_init pthread_attr_init
# define __pthread_attr_setdetachstate pthread_attr_setdetachstate
#endif

#ifndef aio_start_notify_thread
# define aio_start_notify_thread() do { } while (0)
#endif

struct notify_func
  {
    void (*func) (sigval_t);
    sigval_t value;
  };

static void *
notify_func_wrapper (void *arg)
{
  aio_start_notify_thread ();
  struct notify_func *const n = arg;
  void (*func) (sigval_t) = n->func;
  sigval_t value = n->value;
  free (n);
  (*func) (value);
  return NULL;
}


int
__aio_notify_only (struct sigevent *sigev)
{
  int result = 0;

  /* Send the signal to notify about finished processing of the request.  */
  if (__glibc_unlikely (sigev->sigev_notify == SIGEV_THREAD))
    {
      /* We have to start a thread.  */
      pthread_t tid;
      pthread_attr_t attr, *pattr;

      pattr = (pthread_attr_t *) sigev->sigev_notify_attributes;
      if (pattr == NULL)
	{
	  __pthread_attr_init (&attr);
	  __pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
	  pattr = &attr;
	}

      /* SIGEV may be freed as soon as we return, so we cannot let the
	 notification thread use that pointer.  Even though a sigval_t is
	 only one word and the same size as a void *, we cannot just pass
	 the value through pthread_create as the argument and have the new
	 thread run the user's function directly, because on some machines
	 the calling convention for a union like sigval_t is different from
	 that for a pointer type like void *.  */
      struct notify_func *nf = malloc (sizeof *nf);
      if (nf == NULL)
	result = -1;
      else
	{
	  nf->func = sigev->sigev_notify_function;
	  nf->value = sigev->sigev_value;
	  if (__pthread_create (&tid, pattr, notify_func_wrapper, nf) < 0)
	    {
	      free (nf);
	      result = -1;
	    }
	}
    }
  else if (sigev->sigev_notify == SIGEV_SIGNAL)
    {
      /* We have to send a signal.  */
#if _POSIX_REALTIME_SIGNALS > 0
      /* Note that the standard gives us the option of using a plain
	 non-queuing signal here when SA_SIGINFO is not set for the signal.  */
      if (__aio_sigqueue (sigev->sigev_signo, sigev->sigev_value, getpid ())
	  < 0)
	result = -1;
#else
      /* There are no queued signals on this system at all.  */
      result = raise (sigev->sigev_signo);
#endif
    }

  return result;
}


void
__aio_notify (struct requestlist *req)
{
  struct waitlist *waitlist;
  struct aiocb *aiocbp = &req->aiocbp->aiocb;

  if (__aio_notify_only (&aiocbp->aio_sigevent) != 0)
    {
      /* XXX What shall we do if already an error is set by
	 read/write/fsync?  */
      aiocbp->__error_code = errno;
      aiocbp->__return_value = -1;
    }

  /* Now also notify possibly waiting threads.  */
  waitlist = req->waiting;
  while (waitlist != NULL)
    {
      struct waitlist *next = waitlist->next;

      if (waitlist->sigevp == NULL)
	{
	  if (waitlist->result != NULL && aiocbp->__return_value == -1)
	    *waitlist->result = -1;

#ifdef DONT_NEED_AIO_MISC_COND
	  AIO_MISC_NOTIFY (waitlist);
#else
	  /* Decrement the counter.  */
	  --*waitlist->counterp;

	  pthread_cond_signal (waitlist->cond);
#endif
	}
      else
	/* This is part of an asynchronous `lio_listio' operation.  If
	   this request is the last one, send the signal.  */
	if (--*waitlist->counterp == 0)
	  {
	    __aio_notify_only (waitlist->sigevp);
	    /* This is tricky.  See lio_listio.c for the reason why
	       this works.  */
	    free ((void *) waitlist->counterp);
	  }

      waitlist = next;
    }
}
