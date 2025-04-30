/* Enqueue and list of read or write requests.  Common code template.
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

/* The following macros must be defined before including this file:

   LIO_LISTIO         The public symbol (lio_listio or lio_listio64).
   AIOCB              Struct tag used by LIO_LISTIO (aiocb or aiocb64).
   LIO_LISTIO_OLD     The internal symbol for the compat implementation.
   LIO_LISTIO_NEW     The internal symbol for the current implementation.
   LIO_OPCODE_BASE    Opcode shift for 64-bit version with 32-bit word size.

   For __WORDSIZE == 64, LIO_LISTIO must always be lio_listio, and
   lio_listio64 is automatically defined as well.  */

#include <bits/wordsize.h>
#if __WORDSIZE == 64
# define lio_listio64 XXX
# include <aio.h>
/* And undo the hack.  */
# undef lio_listio64
#else
# include <aio.h>
#endif

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthreadP.h>

#include <aio_misc.h>

#include <shlib-compat.h>


/* We need this special structure to handle asynchronous I/O.  */
struct async_waitlist
  {
    unsigned int counter;
    struct sigevent sigev;
    struct waitlist list[0];
  };


/* The code in glibc 2.1 to glibc 2.4 issued only one event when all
   requests submitted with lio_listio finished.  The existing practice
   is to issue events for the individual requests as well.  This is
   what the new code does.  */
#if SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_4)
# define LIO_MODE(mode) ((mode) & 127)
# define NO_INDIVIDUAL_EVENT_P(mode) ((mode) & 128)
#else
# define LIO_MODE(mode) mode
# define NO_INDIVIDUAL_EVENT_P(mode) 0
#endif


static int
lio_listio_internal (int mode, struct AIOCB *const list[], int nent,
		     struct sigevent *sig)
{
  struct sigevent defsigev;
  struct requestlist *requests[nent];
  int cnt;
  volatile unsigned int total = 0;
  int result = 0;

  if (sig == NULL)
    {
      defsigev.sigev_notify = SIGEV_NONE;
      sig = &defsigev;
    }

  /* Request the mutex.  */
  __pthread_mutex_lock (&__aio_requests_mutex);

  /* Now we can enqueue all requests.  Since we already acquired the
     mutex the enqueue function need not do this.  */
  for (cnt = 0; cnt < nent; ++cnt)
    if (list[cnt] != NULL && list[cnt]->aio_lio_opcode != LIO_NOP)
      {
	if (NO_INDIVIDUAL_EVENT_P (mode))
	  list[cnt]->aio_sigevent.sigev_notify = SIGEV_NONE;

	requests[cnt] = __aio_enqueue_request ((aiocb_union *) list[cnt],
					       (list[cnt]->aio_lio_opcode
						| LIO_OPCODE_BASE));

	if (requests[cnt] != NULL)
	  /* Successfully enqueued.  */
	  ++total;
	else
	  /* Signal that we've seen an error.  `errno' and the error code
	     of the aiocb will tell more.  */
	  result = -1;
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
      __pthread_mutex_unlock (&__aio_requests_mutex);

      if (LIO_MODE (mode) == LIO_NOWAIT)
	__aio_notify_only (sig);

      return result;
    }
  else if (LIO_MODE (mode) == LIO_WAIT)
    {
#ifndef DONT_NEED_AIO_MISC_COND
      pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
      int oldstate;
#endif
      struct waitlist waitlist[nent];

      total = 0;
      for (cnt = 0; cnt < nent; ++cnt)
	{
	  assert (requests[cnt] == NULL || list[cnt] != NULL);

	  if (requests[cnt] != NULL && list[cnt]->aio_lio_opcode != LIO_NOP)
	    {
#ifndef DONT_NEED_AIO_MISC_COND
	      waitlist[cnt].cond = &cond;
#endif
	      waitlist[cnt].result = &result;
	      waitlist[cnt].next = requests[cnt]->waiting;
	      waitlist[cnt].counterp = &total;
	      waitlist[cnt].sigevp = NULL;
	      requests[cnt]->waiting = &waitlist[cnt];
	      ++total;
	    }
	}

#ifdef DONT_NEED_AIO_MISC_COND
      AIO_MISC_WAIT (result, total, NULL, 0);
#else
      /* Since `pthread_cond_wait'/`pthread_cond_timedwait' are cancellation
	 points we must be careful.  We added entries to the waiting lists
	 which we must remove.  So defer cancellation for now.  */
      pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &oldstate);

      while (total > 0)
	pthread_cond_wait (&cond, &__aio_requests_mutex);

      /* Now it's time to restore the cancellation state.  */
      pthread_setcancelstate (oldstate, NULL);

      /* Release the conditional variable.  */
      if (pthread_cond_destroy (&cond) != 0)
	/* This must never happen.  */
	abort ();
#endif

      /* If any of the I/O requests failed, return -1 and set errno.  */
      if (result != 0)
	{
	  __set_errno (result == EINTR ? EINTR : EIO);
	  result = -1;
	}
    }
  else
    {
      struct async_waitlist *waitlist;

      waitlist = (struct async_waitlist *)
	malloc (sizeof (struct async_waitlist)
		+ (nent * sizeof (struct waitlist)));

      if (waitlist == NULL)
	{
	  __set_errno (EAGAIN);
	  result = -1;
	}
      else
	{
	  total = 0;

	  for (cnt = 0; cnt < nent; ++cnt)
	    {
	      assert (requests[cnt] == NULL || list[cnt] != NULL);

	      if (requests[cnt] != NULL
		  && list[cnt]->aio_lio_opcode != LIO_NOP)
		{
#ifndef DONT_NEED_AIO_MISC_COND
		  waitlist->list[cnt].cond = NULL;
#endif
		  waitlist->list[cnt].result = NULL;
		  waitlist->list[cnt].next = requests[cnt]->waiting;
		  waitlist->list[cnt].counterp = &waitlist->counter;
		  waitlist->list[cnt].sigevp = &waitlist->sigev;
		  requests[cnt]->waiting = &waitlist->list[cnt];
		  ++total;
		}
	    }

	  waitlist->counter = total;
	  waitlist->sigev = *sig;
	}
    }

  /* Release the mutex.  */
  __pthread_mutex_unlock (&__aio_requests_mutex);

  return result;
}


#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_4)
int
attribute_compat_text_section
LIO_LISTIO_OLD (int mode, struct AIOCB *const list[], int nent,
                struct sigevent *sig)
{
  /* Check arguments.  */
  if (mode != LIO_WAIT && mode != LIO_NOWAIT)
    {
      __set_errno (EINVAL);
      return -1;
    }

  return lio_listio_internal (mode | LIO_NO_INDIVIDUAL_EVENT, list, nent, sig);
}
compat_symbol (librt, LIO_LISTIO_OLD, LIO_LISTIO, GLIBC_2_1);
# if __WORDSIZE == 64
compat_symbol (librt, LIO_LISTIO_OLD, lio_listio64, GLIBC_2_1);
# endif
#endif /* OTHER_SHLIB_COMPAT */


int
LIO_LISTIO_NEW (int mode, struct AIOCB *const list[], int nent,
                struct sigevent *sig)
{
    /* Check arguments.  */
  if (mode != LIO_WAIT && mode != LIO_NOWAIT)
    {
      __set_errno (EINVAL);
      return -1;
    }

  return lio_listio_internal (mode, list, nent, sig);
}

#if PTHREAD_IN_LIBC
versioned_symbol (libc, LIO_LISTIO_NEW, LIO_LISTIO, GLIBC_2_34);
# if __WORDSIZE == 64
versioned_symbol (libc, LIO_LISTIO_NEW, lio_listio64, GLIBC_2_34);
# endif
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_4, GLIBC_2_34)
compat_symbol (librt, LIO_LISTIO_NEW, LIO_LISTIO, GLIBC_2_4);
#  if __WORDSIZE == 64
compat_symbol (librt, LIO_LISTIO_NEW, lio_listio64, GLIBC_2_4);
#  endif
# endif /* OTHER_SHLIB_COMPAT */
#else /* !PTHREAD_IN_LIBC */
versioned_symbol (librt, LIO_LISTIO_NEW, LIO_LISTIO, GLIBC_2_4);
# if __WORDSIZE == 64
versioned_symbol (librt, LIO_LISTIO_NEW, lio_listio64, GLIBC_2_4);
# endif
#endif /* !PTHREAD_IN_LIBC */
