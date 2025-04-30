/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include "pthreadP.h"
#include <atomic.h>
#include <futex-internal.h>
#include <shlib-compat.h>

int
__pthread_barrier_destroy (pthread_barrier_t *barrier)
{
  struct pthread_barrier *bar = (struct pthread_barrier *) barrier;

  /* Destroying a barrier is only allowed if no thread is blocked on it.
     Thus, there is no unfinished round, and all modifications to IN will
     have happened before us (either because the calling thread took part
     in the most recent round and thus synchronized-with all other threads
     entering, or the program ensured this through other synchronization).
     We must wait until all threads that entered so far have confirmed that
     they have exited as well.  To get the notification, pretend that we have
     reached the reset threshold.  */
  unsigned int count = bar->count;
  unsigned int max_in_before_reset = BARRIER_IN_THRESHOLD
				   - BARRIER_IN_THRESHOLD % count;
  /* Relaxed MO sufficient because the program must have ensured that all
     modifications happen-before this load (see above).  */
  unsigned int in = atomic_load_relaxed (&bar->in);
  /* Trigger reset.  The required acquire MO is below.  */
  if (atomic_fetch_add_relaxed (&bar->out, max_in_before_reset - in) < in)
    {
      /* Not all threads confirmed yet that they have exited, so another
	 thread will perform a reset.  Wait until that has happened.  */
      while (in != 0)
	{
	  futex_wait_simple (&bar->in, in, bar->shared);
	  in = atomic_load_relaxed (&bar->in);
	}
    }
  /* We must ensure that memory reuse happens after all prior use of the
     barrier (specifically, synchronize-with the reset of the barrier or the
     confirmation of threads leaving the barrier).  */
  atomic_thread_fence_acquire ();

  return 0;
}
versioned_symbol (libc, __pthread_barrier_destroy, pthread_barrier_destroy,
                  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_barrier_destroy, pthread_barrier_destroy,
               GLIBC_2_2);
#endif
