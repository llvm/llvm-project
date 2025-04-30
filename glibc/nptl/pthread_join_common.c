/* Common definition for pthread_{timed,try}join{_np}.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include "pthreadP.h"
#include <atomic.h>
#include <stap-probe.h>
#include <time.h>
#include <futex-internal.h>

static void
cleanup (void *arg)
{
  /* If we already changed the waiter ID, reset it.  The call cannot
     fail for any reason but the thread not having done that yet so
     there is no reason for a loop.  */
  struct pthread *self = THREAD_SELF;
  atomic_compare_exchange_weak_acquire (&arg, (void**)&self, NULL);
}

int
__pthread_clockjoin_ex (pthread_t threadid, void **thread_return,
                        clockid_t clockid,
                        const struct __timespec64 *abstime, bool block)
{
  struct pthread *pd = (struct pthread *) threadid;

  /* Make sure the descriptor is valid.  */
  if (INVALID_NOT_TERMINATED_TD_P (pd))
    /* Not a valid thread handle.  */
    return ESRCH;

  /* Is the thread joinable?.  */
  if (IS_DETACHED (pd))
    /* We cannot wait for the thread.  */
    return EINVAL;

  struct pthread *self = THREAD_SELF;
  int result = 0;

  LIBC_PROBE (pthread_join, 1, threadid);

  if ((pd == self
       || (self->joinid == pd
	   && (pd->cancelhandling
	       & (CANCELED_BITMASK | EXITING_BITMASK
		  | TERMINATED_BITMASK)) == 0))
      && !(self->cancelstate == PTHREAD_CANCEL_ENABLE
	   && (pd->cancelhandling & (CANCELED_BITMASK | EXITING_BITMASK
				     | TERMINATED_BITMASK))
	       == CANCELED_BITMASK))
    /* This is a deadlock situation.  The threads are waiting for each
       other to finish.  Note that this is a "may" error.  To be 100%
       sure we catch this error we would have to lock the data
       structures but it is not necessary.  In the unlikely case that
       two threads are really caught in this situation they will
       deadlock.  It is the programmer's problem to figure this
       out.  */
    return EDEADLK;

  /* Wait for the thread to finish.  If it is already locked something
     is wrong.  There can only be one waiter.  */
  else if (__glibc_unlikely (atomic_compare_exchange_weak_acquire (&pd->joinid,
								   &self,
								   NULL)))
    /* There is already somebody waiting for the thread.  */
    return EINVAL;

  /* BLOCK waits either indefinitely or based on an absolute time.  POSIX also
     states a cancellation point shall occur for pthread_join, and we use the
     same rationale for posix_timedjoin_np.  Both clockwait_tid and the futex
     call use the cancellable variant.  */
  if (block)
    {
      /* During the wait we change to asynchronous cancellation.  If we
	 are cancelled the thread we are waiting for must be marked as
	 un-wait-ed for again.  */
      pthread_cleanup_push (cleanup, &pd->joinid);

      /* We need acquire MO here so that we synchronize with the
         kernel's store to 0 when the clone terminates. (see above)  */
      pid_t tid;
      while ((tid = atomic_load_acquire (&pd->tid)) != 0)
        {
         /* The kernel notifies a process which uses CLONE_CHILD_CLEARTID via
	    futex wake-up when the clone terminates.  The memory location
	    contains the thread ID while the clone is running and is reset to
	    zero by the kernel afterwards.  The kernel up to version 3.16.3
	    does not use the private futex operations for futex wake-up when
	    the clone terminates.  */
	  int ret = __futex_abstimed_wait_cancelable64 (
	    (unsigned int *) &pd->tid, tid, clockid, abstime, LLL_SHARED);
	  if (ret == ETIMEDOUT || ret == EOVERFLOW)
	    {
	      result = ret;
	      break;
	    }
	}

      pthread_cleanup_pop (0);
    }

  void *pd_result = pd->result;
  if (__glibc_likely (result == 0))
    {
      /* We mark the thread as terminated and as joined.  */
      pd->tid = -1;

      /* Store the return value if the caller is interested.  */
      if (thread_return != NULL)
	*thread_return = pd_result;

      /* Free the TCB.  */
      __nptl_free_tcb (pd);
    }
  else
    pd->joinid = NULL;

  LIBC_PROBE (pthread_join_ret, 3, threadid, result, pd_result);

  return result;
}
