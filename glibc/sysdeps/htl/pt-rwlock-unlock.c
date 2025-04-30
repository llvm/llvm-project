/* Unlock a rwlock.  Generic version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <pt-internal.h>

/* Unlock *RWLOCK, rescheduling a waiting writer thread or, if there
   are no threads waiting for a write lock, rescheduling the reader
   threads.  */
int
__pthread_rwlock_unlock (pthread_rwlock_t *rwlock)
{
  struct __pthread *wakeup;

  __pthread_spin_wait (&rwlock->__lock);

  assert (__pthread_spin_trylock (&rwlock->__held) == EBUSY);

  if (rwlock->__readers > 1)
    /* There are other readers.  */
    {
      rwlock->__readers--;
      __pthread_spin_unlock (&rwlock->__lock);
      return 0;
    }

  if (rwlock->__readers == 1)
    /* Last reader.  */
    rwlock->__readers = 0;


  /* Wake someone else up.  Try the writer queue first, then the
     reader queue if that is empty.  */

  if (rwlock->__writerqueue)
    {
      wakeup = rwlock->__writerqueue;
      __pthread_dequeue (wakeup);

      /* We do not unlock RWLOCK->held: we are transferring the ownership
         to the thread that we are waking up.  */

      __pthread_spin_unlock (&rwlock->__lock);
      __pthread_wakeup (wakeup);

      return 0;
    }

  if (rwlock->__readerqueue)
    {
      unsigned n = 0;

      __pthread_queue_iterate (rwlock->__readerqueue, wakeup)
	n++;

      {
	struct __pthread *wakeups[n];
	unsigned i = 0;

	__pthread_dequeuing_iterate (rwlock->__readerqueue, wakeup)
	  wakeups[i++] = wakeup;

	rwlock->__readers += n;
	rwlock->__readerqueue = 0;

	__pthread_spin_unlock (&rwlock->__lock);

	for (i = 0; i < n; i++)
	  __pthread_wakeup (wakeups[i]);
      }

      return 0;
    }


  /* Noone is waiting.  Just unlock it.  */

  __pthread_spin_unlock (&rwlock->__held);
  __pthread_spin_unlock (&rwlock->__lock);
  return 0;
}
weak_alias (__pthread_rwlock_unlock, pthread_rwlock_unlock);
