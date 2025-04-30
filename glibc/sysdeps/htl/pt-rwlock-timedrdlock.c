/* Acquire a rwlock for reading.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <assert.h>
#include <time.h>

#include <pt-internal.h>

/* Acquire the rwlock *RWLOCK for reading blocking until *ABSTIME if
   it is already held.  As a GNU extension, if TIMESPEC is NULL then
   wait forever.  */
int
__pthread_rwlock_timedrdlock_internal (struct __pthread_rwlock *rwlock,
				       clockid_t clockid,
				       const struct timespec *abstime)
{
  error_t err;
  int drain;
  struct __pthread *self;

  __pthread_spin_wait (&rwlock->__lock);
  if (__pthread_spin_trylock (&rwlock->__held) == 0)
    /* Successfully acquired the lock.  */
    {
      assert (rwlock->__readerqueue == 0);
      assert (rwlock->__writerqueue == 0);
      assert (rwlock->__readers == 0);

      rwlock->__readers = 1;
      __pthread_spin_unlock (&rwlock->__lock);
      return 0;
    }
  else
    /* Lock is held, but is held by a reader?  */
  if (rwlock->__readers > 0)
    /* Just add ourself to number of readers.  */
    {
      assert (rwlock->__readerqueue == 0);
      rwlock->__readers++;
      __pthread_spin_unlock (&rwlock->__lock);
      return 0;
    }

  /* The lock is busy.  */

  /* Better be blocked by a writer.  */
  assert (rwlock->__readers == 0);

  if (abstime != NULL && ! valid_nanoseconds (abstime->tv_nsec))
    {
      __pthread_spin_unlock (&rwlock->__lock);
      return EINVAL;
    }

  self = _pthread_self ();

  /* Add ourself to the queue.  */
  __pthread_enqueue (&rwlock->__readerqueue, self);
  __pthread_spin_unlock (&rwlock->__lock);

  /* Block the thread.  */
  if (abstime != NULL)
    err = __pthread_timedblock (self, abstime, clockid);
  else
    {
      err = 0;
      __pthread_block (self);
    }

  __pthread_spin_wait (&rwlock->__lock);
  if (self->prevp == NULL)
    /* Another thread removed us from the queue, which means a wakeup message
       has been sent.  It was either consumed while we were blocking, or
       queued after we timed out and before we acquired the rwlock lock, in
       which case the message queue must be drained.  */
    drain = err ? 1 : 0;
  else
    {
      /* We're still in the queue.  Noone attempted to wake us up, i.e. we
         timed out.  */
      __pthread_dequeue (self);
      drain = 0;
    }
  __pthread_spin_unlock (&rwlock->__lock);

  if (drain)
    __pthread_block (self);

  if (err)
    {
      assert (err == ETIMEDOUT);
      return err;
    }

  /* The reader count has already been increment by whoever woke us
     up.  */

  assert (rwlock->__readers > 0);

  return 0;
}

int
__pthread_rwlock_timedrdlock (struct __pthread_rwlock *rwlock,
			      const struct timespec *abstime)
{
  return __pthread_rwlock_timedrdlock_internal (rwlock, CLOCK_REALTIME, abstime);
}
weak_alias (__pthread_rwlock_timedrdlock, pthread_rwlock_timedrdlock)

int
__pthread_rwlock_clockrdlock (struct __pthread_rwlock *rwlock,
			      clockid_t clockid,
			      const struct timespec *abstime)
{
  return __pthread_rwlock_timedrdlock_internal (rwlock, clockid, abstime);
}
weak_alias (__pthread_rwlock_clockrdlock, pthread_rwlock_clockrdlock)
