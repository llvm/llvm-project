/* pthread_barrier_wait.  Generic version.
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

#include <pt-internal.h>

int
pthread_barrier_wait (pthread_barrier_t *barrier)
{
  __pthread_spin_wait (&barrier->__lock);
  if (--barrier->__pending == 0)
    {
      barrier->__pending = barrier->__count;

      if (barrier->__count == 1)
	__pthread_spin_unlock (&barrier->__lock);
      else
	{
	  struct __pthread *wakeup;
	  unsigned n = 0;

	  __pthread_queue_iterate (barrier->__queue, wakeup)
	    n++;

	  {
	    struct __pthread *wakeups[n];
	    unsigned i = 0;

	    __pthread_dequeuing_iterate (barrier->__queue, wakeup)
	      wakeups[i++] = wakeup;

	    barrier->__queue = NULL;
	    __pthread_spin_unlock (&barrier->__lock);

	    for (i = 0; i < n; i++)
	      __pthread_wakeup (wakeups[i]);
	  }
	}

      return PTHREAD_BARRIER_SERIAL_THREAD;
    }
  else
    {
      struct __pthread *self = _pthread_self ();

      /* Add ourselves to the list of waiters.  */
      __pthread_enqueue (&barrier->__queue, self);
      __pthread_spin_unlock (&barrier->__lock);

      __pthread_block (self);
      return 0;
    }
}
