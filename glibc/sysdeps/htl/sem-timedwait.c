/* Wait on a semaphore with a timeout.  Generic version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <semaphore.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include <hurdlock.h>
#include <hurd/hurd.h>
#include <sysdep-cancel.h>

#include <pt-internal.h>

#if !__HAVE_64B_ATOMICS
static void
__sem_wait_32_finish (struct new_sem *isem);
#endif

static void
__sem_wait_cleanup (void *arg)
{
  struct new_sem *isem = arg;

#if __HAVE_64B_ATOMICS
  atomic_fetch_add_relaxed (&isem->data, -((uint64_t) 1 << SEM_NWAITERS_SHIFT));
#else
  __sem_wait_32_finish (isem);
#endif
}

int
__sem_timedwait_internal (sem_t *restrict sem,
			  clockid_t clock_id,
			  const struct timespec *restrict timeout)
{
  struct new_sem *isem = (struct new_sem *) sem;
  int err, ret = 0;
  int flags = isem->pshared ? GSYNC_SHARED : 0;

  __pthread_testcancel ();

  if (__sem_waitfast (isem, 0) == 0)
    return 0;

  int cancel_oldtype = LIBC_CANCEL_ASYNC();

#if __HAVE_64B_ATOMICS
  uint64_t d = atomic_fetch_add_relaxed (&sem->data,
		 (uint64_t) 1 << SEM_NWAITERS_SHIFT);

  pthread_cleanup_push (__sem_wait_cleanup, isem);

  for (;;)
    {
      if ((d & SEM_VALUE_MASK) == 0)
	{
	  /* No token, sleep.  */
	  if (timeout)
	    err = __lll_abstimed_wait_intr (
		      ((unsigned int *) &sem->data) + SEM_VALUE_OFFSET,
		      0, timeout, flags, clock_id);
	  else
	    err = __lll_wait_intr (
		      ((unsigned int *) &sem->data) + SEM_VALUE_OFFSET,
		      0, flags);

	  if (err != 0)
	    {
	      /* Error, interruption or timeout, abort.  */
	      if (err == KERN_TIMEDOUT)
		err = ETIMEDOUT;
	      if (err == KERN_INTERRUPTED)
		err = EINTR;
	      ret = __hurd_fail (err);
	      __sem_wait_cleanup (isem);
	      break;
	    }

	  /* Token changed */
	  d = atomic_load_relaxed (&sem->data);
	}
      else
	{
	  /* Try to acquire and dequeue.  */
	  if (atomic_compare_exchange_weak_acquire (&sem->data,
	      &d, d - 1 - ((uint64_t) 1 << SEM_NWAITERS_SHIFT)))
	    {
	      /* Success */
	      ret = 0;
	      break;
	    }
	}
    }

  pthread_cleanup_pop (0);
#else
  unsigned int v;

  atomic_fetch_add_acquire (&isem->nwaiters, 1);

  pthread_cleanup_push (__sem_wait_cleanup, isem);

  v = atomic_load_relaxed (&isem->value);
  do
    {
      do
	{
	  do
	    {
	      if ((v & SEM_NWAITERS_MASK) != 0)
		break;
	    }
	  while (!atomic_compare_exchange_weak_release (&isem->value,
	      &v, v | SEM_NWAITERS_MASK));

	  if ((v >> SEM_VALUE_SHIFT) == 0)
	    {
	      /* No token, sleep.  */
	      if (timeout)
		err = __lll_abstimed_wait_intr (&isem->value,
			  SEM_NWAITERS_MASK, timeout, flags, clock_id);
	      else
		err = __lll_wait_intr (&isem->value,
			  SEM_NWAITERS_MASK, flags);

	      if (err != 0)
		{
		  /* Error, interruption or timeout, abort.  */
		  if (err == KERN_TIMEDOUT)
		    err = ETIMEDOUT;
		  if (err == KERN_INTERRUPTED)
		    err = EINTR;
		  ret = __hurd_fail (err);
		  goto error;
		}

	      /* Token changed */
	      v = atomic_load_relaxed (&isem->value);
	    }
	}
      while ((v >> SEM_VALUE_SHIFT) == 0);
    }
  while (!atomic_compare_exchange_weak_acquire (&isem->value,
	  &v, v - (1 << SEM_VALUE_SHIFT)));

error:
  pthread_cleanup_pop (0);

  __sem_wait_32_finish (isem);
#endif

  LIBC_CANCEL_RESET (cancel_oldtype);

  return ret;
}

#if !__HAVE_64B_ATOMICS
/* Stop being a registered waiter (non-64b-atomics code only).  */
static void
__sem_wait_32_finish (struct new_sem *isem)
{
  unsigned int wguess = atomic_load_relaxed (&isem->nwaiters);
  if (wguess == 1)
    atomic_fetch_and_acquire (&isem->value, ~SEM_NWAITERS_MASK);

  unsigned int wfinal = atomic_fetch_add_release (&isem->nwaiters, -1);
  if (wfinal > 1 && wguess == 1)
    {
      unsigned int v = atomic_fetch_or_relaxed (&isem->value,
						SEM_NWAITERS_MASK);
      v >>= SEM_VALUE_SHIFT;
      while (v--)
	__lll_wake (&isem->value, isem->pshared ? GSYNC_SHARED : 0);
    }
}
#endif

int
__sem_clockwait (sem_t *sem, clockid_t clockid,
		 const struct timespec *restrict timeout)
{
  return __sem_timedwait_internal (sem, clockid, timeout);
}
weak_alias (__sem_clockwait, sem_clockwait);

int
__sem_timedwait (sem_t *restrict sem, const struct timespec *restrict timeout)
{
  return __sem_timedwait_internal (sem, CLOCK_REALTIME, timeout);
}

weak_alias (__sem_timedwait, sem_timedwait);
