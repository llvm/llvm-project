/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky <schwidefsky@de.ibm.com>, 2003.

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

#include <errno.h>
#include <sysdep.h>
#include <futex-internal.h>
#include <pthreadP.h>
#include <shlib-compat.h>


/* Wait on the barrier.

   In each round, we wait for a fixed number of threads to enter the barrier
   (COUNT).  Once that has happened, exactly these threads are allowed to
   leave the barrier.  Note that POSIX does not require that only COUNT
   threads can attempt to block using the barrier concurrently.

   We count the number of threads that have entered (IN).  Each thread
   increments IN when entering, thus getting a position in the sequence of
   threads that are or have been waiting (starting with 1, so the position
   is the number of threads that have entered so far including the current
   thread).
   CURRENT_ROUND designates the most recent thread whose round has been
   detected as complete.  When a thread detects that enough threads have
   entered to make a round complete, it finishes this round by effectively
   adding COUNT to CURRENT_ROUND atomically.  Threads that believe that their
   round is not complete yet wait until CURRENT_ROUND is not smaller than
   their position anymore.

   A barrier can be destroyed as soon as no threads are blocked on the
   barrier.  This is already the case if just one thread from the last round
   has stopped waiting and returned to the caller; the assumption is that
   all threads from the round are unblocked atomically, even though they may
   return at different times from the respective calls to
   pthread_barrier_wait).  Thus, a valid call to pthread_barrier_destroy can
   be concurrent with other threads still figuring out that their round has
   been completed.  Therefore, threads need to confirm that they have left
   the barrier by incrementing OUT, and pthread_barrier_destroy needs to wait
   until OUT equals IN.

   To avoid an ABA issue for futex_wait on CURRENT_ROUND and for archs with
   32b-only atomics, we additionally reset the barrier when IN reaches
   a threshold to avoid overflow.  We assume that the total number of threads
   is less than UINT_MAX/2, and set the threshold accordingly so that we can
   use a simple atomic_fetch_add on IN instead of a CAS when entering.  The
   threshold is always set to the end of a round, so all threads that have
   entered are either pre-reset threads or post-reset threads (i.e., have a
   position larger than the threshold).
   Pre-reset threads just run the algorithm explained above.  Post-reset
   threads wait until IN is reset to a pre-threshold value.
   When the last pre-reset thread leaves the barrier (i.e., OUT equals the
   threshold), it resets the barrier to its initial state.  Other (post-reset)
   threads wait for the reset to have finished by waiting until IN is less
   than the threshold and then restart by trying to enter the barrier again.

   We reuse the reset mechanism in pthread_barrier_destroy to get notified
   when all threads have left the barrier: We trigger an artificial reset and
   wait for the last pre-reset thread to finish reset, thus notifying the
   thread that is about to destroy the barrier.

   Blocking using futexes is straightforward: pre-reset threads wait for
   completion of their round using CURRENT_ROUND as futex word, and post-reset
   threads and pthread_barrier_destroy use IN as futex word.

   Further notes:
   * It is not simple to let some of the post-reset threads help with the
     reset because of the ABA issues that arise; therefore, we simply make
     the last thread to leave responsible for the reset.
   * POSIX leaves it unspecified whether a signal handler running in a thread
     that has been unblocked (because its round is complete) can stall all
     other threads and prevent them from returning from the barrier.  In this
     implementation, other threads will return.  However,
     pthread_barrier_destroy will of course wait for the signal handler thread
     to confirm that it left the barrier.

   TODO We should add spinning with back-off.  Once we do that, we could also
   try to avoid the futex_wake syscall when a round is detected as finished.
   If we do not spin, it is quite likely that at least some other threads will
   have called futex_wait already.  */
int
___pthread_barrier_wait (pthread_barrier_t *barrier)
{
  struct pthread_barrier *bar = (struct pthread_barrier *) barrier;

  /* How many threads entered so far, including ourself.  */
  unsigned int i;

 reset_restart:
  /* Try to enter the barrier.  We need acquire MO to (1) ensure that if we
     observe that our round can be completed (see below for our attempt to do
     so), all pre-barrier-entry effects of all threads in our round happen
     before us completing the round, and (2) to make our use of the barrier
     happen after a potential reset.  We need release MO to make sure that our
     pre-barrier-entry effects happen before threads in this round leaving the
     barrier.  */
  i = atomic_fetch_add_acq_rel (&bar->in, 1) + 1;
  /* These loads are after the fetch_add so that we're less likely to first
     pull in the cache line as shared.  */
  unsigned int count = bar->count;
  /* This is the number of threads that can enter before we need to reset.
     Always at the end of a round.  */
  unsigned int max_in_before_reset = BARRIER_IN_THRESHOLD
				   - BARRIER_IN_THRESHOLD % count;

  if (i > max_in_before_reset)
    {
      /* We're in a reset round.  Just wait for a reset to finish; do not
	 help finishing previous rounds because this could happen
	 concurrently with a reset.  */
      while (i > max_in_before_reset)
	{
	  futex_wait_simple (&bar->in, i, bar->shared);
	  /* Relaxed MO is fine here because we just need an indication for
	     when we should retry to enter (which will use acquire MO, see
	     above).  */
	  i = atomic_load_relaxed (&bar->in);
	}
      goto reset_restart;
    }

  /* Look at the current round.  At this point, we are just interested in
     whether we can complete rounds, based on the information we obtained
     through our acquire-MO load of IN.  Nonetheless, if we notice that
     our round has been completed using this load, we use the acquire-MO
     fence below to make sure that all pre-barrier-entry effects of all
     threads in our round happen before us leaving the barrier.  Therefore,
     relaxed MO is sufficient.  */
  unsigned cr = atomic_load_relaxed (&bar->current_round);

  /* Try to finish previous rounds and/or the current round.  We simply
     consider just our position here and do not try to do the work of threads
     that entered more recently.  */
  while (cr + count <= i)
    {
      /* Calculate the new current round based on how many threads entered.
	 NEWCR must be larger than CR because CR+COUNT ends a round.  */
      unsigned int newcr = i - i % count;
      /* Try to complete previous and/or the current round.  We need release
	 MO to propagate the happens-before that we observed through reading
	 with acquire MO from IN to other threads.  If the CAS fails, it
	 is like the relaxed-MO load of CURRENT_ROUND above.  */
      if (atomic_compare_exchange_weak_release (&bar->current_round, &cr,
						newcr))
	{
	  /* Update CR with the modification we just did.  */
	  cr = newcr;
	  /* Wake threads belonging to the rounds we just finished.  We may
	     wake more threads than necessary if more than COUNT threads try
	     to block concurrently on the barrier, but this is not a typical
	     use of barriers.
	     Note that we can still access SHARED because we haven't yet
	     confirmed to have left the barrier.  */
	  futex_wake (&bar->current_round, INT_MAX, bar->shared);
	  /* We did as much as we could based on our position.  If we advanced
	     the current round to a round sufficient for us, do not wait for
	     that to happen and skip the acquire fence (we already
	     synchronize-with all other threads in our round through the
	     initial acquire MO fetch_add of IN.  */
	  if (i <= cr)
	    goto ready_to_leave;
	  else
	    break;
	}
    }

  /* Wait until the current round is more recent than the round we are in.  */
  while (i > cr)
    {
      /* Wait for the current round to finish.  */
      futex_wait_simple (&bar->current_round, cr, bar->shared);
      /* See the fence below.  */
      cr = atomic_load_relaxed (&bar->current_round);
    }

  /* Our round finished.  Use the acquire MO fence to synchronize-with the
     thread that finished the round, either through the initial load of
     CURRENT_ROUND above or a failed CAS in the loop above.  */
  atomic_thread_fence_acquire ();

  /* Now signal that we left.  */
  unsigned int o;
 ready_to_leave:
  /* We need release MO here so that our use of the barrier happens before
     reset or memory reuse after pthread_barrier_destroy.  */
  o = atomic_fetch_add_release (&bar->out, 1) + 1;
  if (o == max_in_before_reset)
    {
      /* Perform a reset if we are the last pre-reset thread leaving.   All
	 other threads accessing the barrier are post-reset threads and are
	 incrementing or spinning on IN.  Thus, resetting IN as the last step
	 of reset ensures that the reset is not concurrent with actual use of
	 the barrier.  We need the acquire MO fence so that the reset happens
	 after use of the barrier by all earlier pre-reset threads.  */
      atomic_thread_fence_acquire ();
      atomic_store_relaxed (&bar->current_round, 0);
      atomic_store_relaxed (&bar->out, 0);
      /* When destroying the barrier, we wait for a reset to happen.  Thus,
	 we must load SHARED now so that this happens before the barrier is
	 destroyed.  */
      int shared = bar->shared;
      atomic_store_release (&bar->in, 0);
      futex_wake (&bar->in, INT_MAX, shared);

    }

  /* Return a special value for exactly one thread per round.  */
  return i % count == 0 ?  PTHREAD_BARRIER_SERIAL_THREAD : 0;
}
versioned_symbol (libc, ___pthread_barrier_wait, pthread_barrier_wait,
                  GLIBC_2_34);
libc_hidden_ver (___pthread_barrier_wait, __pthread_barrier_wait)
#ifndef SHARED
strong_alias (___pthread_barrier_wait, __pthread_barrier_wait)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_barrier_wait, pthread_barrier_wait,
               GLIBC_2_2);
#endif
