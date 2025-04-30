/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include "pthreadP.h"
#include <futex-internal.h>
#include <atomic.h>
#include <libc-lockP.h>
#include <shlib-compat.h>

unsigned long int __fork_generation attribute_hidden;


static void
clear_once_control (void *arg)
{
  pthread_once_t *once_control = (pthread_once_t *) arg;

  /* Reset to the uninitialized state here.  We don't need a stronger memory
     order because we do not need to make any other of our writes visible to
     other threads that see this value: This function will be called if we
     get interrupted (see __pthread_once), so all we need to relay to other
     threads is the state being reset again.  */
  atomic_store_relaxed (once_control, 0);
  futex_wake ((unsigned int *) once_control, INT_MAX, FUTEX_PRIVATE);
}


/* This is similar to a lock implementation, but we distinguish between three
   states: not yet initialized (0), initialization in progress
   (__fork_generation | __PTHREAD_ONCE_INPROGRESS), and initialization
   finished (__PTHREAD_ONCE_DONE); __fork_generation does not use the bits
   that are used for __PTHREAD_ONCE_INPROGRESS and __PTHREAD_ONCE_DONE (which
   is what __PTHREAD_ONCE_FORK_GEN_INCR is used for).  If in the first state,
   threads will try to run the initialization by moving to the second state;
   the first thread to do so via a CAS on once_control runs init_routine,
   other threads block.
   When forking the process, some threads can be interrupted during the second
   state; they won't be present in the forked child, so we need to restart
   initialization in the child.  To distinguish an in-progress initialization
   from an interrupted initialization (in which case we need to reclaim the
   lock), we look at the fork generation that's part of the second state: We
   can reclaim iff it differs from the current fork generation.
   XXX: This algorithm has an ABA issue on the fork generation: If an
   initialization is interrupted, we then fork 2^30 times (30 bits of
   once_control are used for the fork generation), and try to initialize
   again, we can deadlock because we can't distinguish the in-progress and
   interrupted cases anymore.
   XXX: We split out this slow path because current compilers do not generate
   as efficient code when the fast path in __pthread_once below is not in a
   separate function.  */
static int
__attribute__ ((noinline))
__pthread_once_slow (pthread_once_t *once_control, void (*init_routine) (void))
{
  while (1)
    {
      int val, newval;

      /* We need acquire memory order for this load because if the value
         signals that initialization has finished, we need to see any
         data modifications done during initialization.  */
      val = atomic_load_acquire (once_control);
      do
	{
	  /* Check if the initialization has already been done.  */
	  if (__glibc_likely ((val & __PTHREAD_ONCE_DONE) != 0))
	    return 0;

	  /* We try to set the state to in-progress and having the current
	     fork generation.  We don't need atomic accesses for the fork
	     generation because it's immutable in a particular process, and
	     forked child processes start with a single thread that modified
	     the generation.  */
	  newval = __fork_generation | __PTHREAD_ONCE_INPROGRESS;
	  /* We need acquire memory order here for the same reason as for the
	     load from once_control above.  */
	}
      while (__glibc_unlikely (!atomic_compare_exchange_weak_acquire (
	  once_control, &val, newval)));

      /* Check if another thread already runs the initializer.	*/
      if ((val & __PTHREAD_ONCE_INPROGRESS) != 0)
	{
	  /* Check whether the initializer execution was interrupted by a
	     fork.  We know that for both values, __PTHREAD_ONCE_INPROGRESS
	     is set and __PTHREAD_ONCE_DONE is not.  */
	  if (val == newval)
	    {
	      /* Same generation, some other thread was faster.  Wait and
		 retry.  */
	      futex_wait_simple ((unsigned int *) once_control,
				 (unsigned int) newval, FUTEX_PRIVATE);
	      continue;
	    }
	}

      /* This thread is the first here.  Do the initialization.
	 Register a cleanup handler so that in case the thread gets
	 interrupted the initialization can be restarted.  */
      pthread_cleanup_combined_push (clear_once_control, once_control);

      init_routine ();

      pthread_cleanup_combined_pop (0);


      /* Mark *once_control as having finished the initialization.  We need
         release memory order here because we need to synchronize with other
         threads that want to use the initialized data.  */
      atomic_store_release (once_control, __PTHREAD_ONCE_DONE);

      /* Wake up all other threads.  */
      futex_wake ((unsigned int *) once_control, INT_MAX, FUTEX_PRIVATE);
      break;
    }

  return 0;
}

int
___pthread_once (pthread_once_t *once_control, void (*init_routine) (void))
{
  /* Fast path.  See __pthread_once_slow.  */
  int val;
  val = atomic_load_acquire (once_control);
  if (__glibc_likely ((val & __PTHREAD_ONCE_DONE) != 0))
    return 0;
  else
    return __pthread_once_slow (once_control, init_routine);
}
libc_hidden_ver (___pthread_once, __pthread_once)
#ifndef SHARED
strong_alias (___pthread_once, __pthread_once)
#endif

versioned_symbol (libc, ___pthread_once, pthread_once, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_once, __pthread_once, GLIBC_2_0);
compat_symbol (libpthread, ___pthread_once, pthread_once, GLIBC_2_0);
#endif
