/* POSIX reader--writer lock: core parts.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#include <pthread.h>
#include <pthreadP.h>
#include <sys/time.h>
#include <stap-probe.h>
#include <atomic.h>
#include <futex-internal.h>
#include <time.h>


/* A reader--writer lock that fulfills the POSIX requirements (but operations
   on this lock are not necessarily full barriers, as one may interpret the
   POSIX requirement about "synchronizing memory").  All critical sections are
   in a total order, writers synchronize with prior writers and readers, and
   readers synchronize with prior writers.

   A thread is allowed to acquire a read lock recursively (i.e., have rdlock
   critical sections that overlap in sequenced-before) unless the kind of the
   rwlock is set to PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP.

   This lock is built so that workloads of mostly readers can be executed with
   low runtime overheads.  This matches that the default kind of the lock is
   PTHREAD_RWLOCK_PREFER_READER_NP.  Acquiring a read lock requires a single
   atomic addition if the lock is or was previously acquired by other
   readers; releasing the lock is a single CAS if there are no concurrent
   writers.
   Workloads consisting of mostly writers are of secondary importance.
   An uncontended write lock acquisition is as fast as for a normal
   exclusive mutex but writer contention is somewhat more costly due to
   keeping track of the exact number of writers.  If the rwlock kind requests
   writers to be preferred (i.e., PTHREAD_RWLOCK_PREFER_WRITER_NP or the
   no-recursive-readers variant of it), then writer--to--writer lock ownership
   hand-over is fairly fast and bypasses lock acquisition attempts by readers.
   The costs of lock ownership transfer between readers and writers vary.  If
   the program asserts that there are no recursive readers and writers are
   preferred, then write lock acquisition attempts will block subsequent read
   lock acquisition attempts, so that new incoming readers do not prolong a
   phase in which readers have acquired the lock.

   The main components of the rwlock are a writer-only lock that allows only
   one of the concurrent writers to be the primary writer, and a
   single-writer-multiple-readers lock that decides between read phases, in
   which readers have acquired the rwlock, and write phases in which a primary
   writer or a sequence of different primary writers have acquired the rwlock.

   The single-writer-multiple-readers lock is the central piece of state
   describing the rwlock and is encoded in the __readers field (see below for
   a detailed explanation):

   State WP  WL  R   RW  Notes
   ---------------------------
   #1    0   0   0   0   Lock is idle (and in a read phase).
   #2    0   0   >0  0   Readers have acquired the lock.
   #3    0   1   0   0   Lock is not acquired; a writer will try to start a
			 write phase.
   #4    0   1   >0  0   Readers have acquired the lock; a writer is waiting
			 and explicit hand-over to the writer is required.
   #4a   0   1   >0  1   Same as #4 except that there are further readers
			 waiting because the writer is to be preferred.
   #5    1   0   0   0   Lock is idle (and in a write phase).
   #6    1   0   >0  0   Write phase; readers will try to start a read phase
			 (requires explicit hand-over to all readers that
			 do not start the read phase).
   #7    1   1   0   0   Lock is acquired by a writer.
   #8    1   1   >0  0   Lock acquired by a writer and readers are waiting;
			 explicit hand-over to the readers is required.

   WP (PTHREAD_RWLOCK_WRPHASE) is true if the lock is in a write phase, so
   potentially acquired by a primary writer.
   WL (PTHREAD_RWLOCK_WRLOCKED) is true if there is a primary writer (i.e.,
   the thread that was able to set this bit from false to true).
   R (all bits in __readers except the number of least-significant bits
   denoted in PTHREAD_RWLOCK_READER_SHIFT) is the number of readers that have
   or are trying to acquired the lock.  There may be more readers waiting if
   writers are preferred and there will be no recursive readers, in which
   case RW (PTHREAD_RWLOCK_RWAITING) is true in state #4a.

   We want to block using futexes but using __readers as a futex word directly
   is not a good solution.  First, we want to wait on different conditions
   such as waiting for a phase change vs. waiting for the primary writer to
   release the writer-only lock.  Second, the number of readers could change
   frequently, which would make it likely that a writer's futex_wait fails
   frequently too because the expected value does not match the value of
   __readers anymore.
   Therefore, we split out the futex words into the __wrphase_futex and
   __writers_futex fields.  The former tracks the value of the WP bit and is
   changed after changing WP by the thread that changes WP.  However, because
   of the POSIX requirements regarding mutex/rwlock destruction (i.e., that
   destroying a rwlock is allowed as soon as no thread has acquired or will
   acquire the lock), we have to be careful and hand over lock ownership (via
   a phase change) carefully to those threads waiting.  Specifically, we must
   prevent a situation in which we are not quite sure whether we still have
   to unblock another thread through a change to memory (executing a
   futex_wake on a former futex word that is now used for something else is
   fine).
   The scheme we use for __wrphase_futex is that waiting threads that may
   use the futex word to block now all have to use the futex word to block; it
   is not allowed to take the short-cut and spin-wait on __readers because
   then the waking thread cannot just make one final change to memory to
   unblock all potentially waiting threads.  If, for example, a reader
   increments R in states #7 or #8, it has to then block until __wrphase_futex
   is 0 and it can confirm that the value of 0 was stored by the primary
   writer; in turn, the primary writer has to change to a read phase too when
   releasing WL (i.e., to state #2), and it must change __wrphase_futex to 0
   as the next step.  This ensures that the waiting reader will not be able to
   acquire, release, and then destroy the lock concurrently with the pending
   futex unblock operations by the former primary writer.  This scheme is
   called explicit hand-over in what follows.
   Note that waiting threads can cancel waiting only if explicit hand-over has
   not yet started (e.g., if __readers is still in states #7 or #8 in the
   example above).

   Writers determine the primary writer through WL.  Blocking using futexes
   is performed using __writers_futex as a futex word; primary writers will
   enable waiting on this futex by setting it to 1 after they acquired the WL
   bit and will disable waiting by setting it to 0 before they release WL.
   This leaves small windows where blocking using futexes is not possible
   although a primary writer exists, but in turn decreases complexity of the
   writer--writer synchronization and does not affect correctness.
   If writers are preferred, writers can hand over WL directly to other
   waiting writers that registered by incrementing __writers:  If the primary
   writer can CAS __writers from a non-zero value to the same value with the
   PTHREAD_RWLOCK_WRHANDOVER bit set, it effectively transfers WL ownership
   to one of the registered waiting writers and does not reset WL; in turn,
   a registered writer that can clear PTHREAD_RWLOCK_WRHANDOVER using a CAS
   then takes over WL.  Note that registered waiting writers can cancel
   waiting by decrementing __writers, but the last writer to unregister must
   become the primary writer if PTHREAD_RWLOCK_WRHANDOVER is set.
   Also note that adding another state/bit to signal potential writer--writer
   contention (e.g., as done in the normal mutex algorithm) would not be
   helpful because we would have to conservatively assume that there is in
   fact no other writer, and wake up readers too.

   To avoid having to call futex_wake when no thread uses __wrphase_futex or
   __writers_futex, threads will set the PTHREAD_RWLOCK_FUTEX_USED bit in the
   respective futex words before waiting on it (using a CAS so it will only be
   set if in a state in which waiting would be possible).  In the case of
   __writers_futex, we wake only one thread but several threads may share
   PTHREAD_RWLOCK_FUTEX_USED, so we must assume that there are still others.
   This is similar to what we do in pthread_mutex_lock.  We do not need to
   do this for __wrphase_futex because there, we always wake all waiting
   threads.

   Blocking in the state #4a simply uses __readers as futex word.  This
   simplifies the algorithm but suffers from some of the drawbacks discussed
   before, though not to the same extent because R can only decrease in this
   state, so the number of potentially failing futex_wait attempts will be
   bounded.  All threads moving from state #4a to another state must wake
   up threads blocked on the __readers futex.

   The ordering invariants that we have to take care of in the implementation
   are primarily those necessary for a reader--writer lock; this is rather
   straightforward and happens during write/read phase switching (potentially
   through explicit hand-over), and between writers through synchronization
   involving the PTHREAD_RWLOCK_WRLOCKED or PTHREAD_RWLOCK_WRHANDOVER bits.
   Additionally, we need to take care that modifications of __writers_futex
   and __wrphase_futex (e.g., by otherwise unordered readers) take place in
   the writer critical sections or read/write phases, respectively, and that
   explicit hand-over observes stores from the previous phase.  How this is
   done is explained in more detail in comments in the code.

   Many of the accesses to the futex words just need relaxed MO.  This is
   possible because we essentially drive both the core rwlock synchronization
   and the futex synchronization in parallel.  For example, an unlock will
   unlock the rwlock and take part in the futex synchronization (using
   PTHREAD_RWLOCK_FUTEX_USED, see above); even if they are not tightly
   ordered in some way, the futex synchronization ensures that there are no
   lost wake-ups, and woken threads will then eventually see the most recent
   state of the rwlock.  IOW, waiting threads will always be woken up, while
   not being able to wait using futexes (which can happen) is harmless; in
   turn, this means that waiting threads don't need special ordering wrt.
   waking threads.

   The futex synchronization consists of the three-state futex word:
   (1) cannot block on it, (2) can block on it, and (3) there might be a
   thread blocked on it (i.e., with PTHREAD_RWLOCK_FUTEX_USED set).
   Relaxed-MO atomic read-modify-write operations are sufficient to maintain
   this (e.g., using a CAS to go from (2) to (3) but not from (1) to (3)),
   but we need ordering of the futex word modifications by the waking threads
   so that they collectively make correct state changes between (1)-(3).
   The futex-internal synchronization (i.e., the conceptual critical sections
   around futex operations in the kernel) then ensures that even an
   unconstrained load (i.e., relaxed MO) inside of futex_wait will not lead to
   lost wake-ups because either the waiting thread will see the change from
   (3) to (1) when a futex_wake came first, or this futex_wake will wake this
   waiting thread because the waiting thread came first.


   POSIX allows but does not require rwlock acquisitions to be a cancellation
   point.  We do not support cancellation.

   TODO We do not try to elide any read or write lock acquisitions currently.
   While this would be possible, it is unclear whether HTM performance is
   currently predictable enough and our runtime tuning is good enough at
   deciding when to use elision so that enabling it would lead to consistently
   better performance.  */


static int
__pthread_rwlock_get_private (pthread_rwlock_t *rwlock)
{
  return rwlock->__data.__shared != 0 ? FUTEX_SHARED : FUTEX_PRIVATE;
}

static __always_inline void
__pthread_rwlock_rdunlock (pthread_rwlock_t *rwlock)
{
  int private = __pthread_rwlock_get_private (rwlock);
  /* We decrease the number of readers, and if we are the last reader and
     there is a primary writer, we start a write phase.  We use a CAS to
     make this atomic so that it is clear whether we must hand over ownership
     explicitly.  */
  unsigned int r = atomic_load_relaxed (&rwlock->__data.__readers);
  unsigned int rnew;
  for (;;)
    {
      rnew = r - (1 << PTHREAD_RWLOCK_READER_SHIFT);
      /* If we are the last reader, we also need to unblock any readers
	 that are waiting for a writer to go first (PTHREAD_RWLOCK_RWAITING)
	 so that they can register while the writer is active.  */
      if ((rnew >> PTHREAD_RWLOCK_READER_SHIFT) == 0)
	{
	  if ((rnew & PTHREAD_RWLOCK_WRLOCKED) != 0)
	    rnew |= PTHREAD_RWLOCK_WRPHASE;
	  rnew &= ~(unsigned int) PTHREAD_RWLOCK_RWAITING;
	}
      /* We need release MO here for three reasons.  First, so that we
	 synchronize with subsequent writers.  Second, we might have been the
	 first reader and set __wrphase_futex to 0, so we need to synchronize
	 with the last reader that will set it to 1 (note that we will always
	 change __readers before the last reader, or we are the last reader).
	 Third, a writer that takes part in explicit hand-over needs to see
	 the first reader's store to __wrphase_futex (or a later value) if
	 the writer observes that a write phase has been started.  */
      if (atomic_compare_exchange_weak_release (&rwlock->__data.__readers,
						&r, rnew))
	break;
      /* TODO Back-off.  */
    }
  if ((rnew & PTHREAD_RWLOCK_WRPHASE) != 0)
    {
      /* We need to do explicit hand-over.  We need the acquire MO fence so
	 that our modification of _wrphase_futex happens after a store by
	 another reader that started a read phase.  Relaxed MO is sufficient
	 for the modification of __wrphase_futex because it is just used
	 to delay acquisition by a writer until all threads are unblocked
	 irrespective of whether they are looking at __readers or
	 __wrphase_futex; any other synchronizes-with relations that are
	 necessary are established through __readers.  */
      atomic_thread_fence_acquire ();
      if ((atomic_exchange_relaxed (&rwlock->__data.__wrphase_futex, 1)
	   & PTHREAD_RWLOCK_FUTEX_USED) != 0)
	futex_wake (&rwlock->__data.__wrphase_futex, INT_MAX, private);
    }
  /* Also wake up waiting readers if we did reset the RWAITING flag.  */
  if ((r & PTHREAD_RWLOCK_RWAITING) != (rnew & PTHREAD_RWLOCK_RWAITING))
    futex_wake (&rwlock->__data.__readers, INT_MAX, private);
}


static __always_inline int
__pthread_rwlock_rdlock_full64 (pthread_rwlock_t *rwlock, clockid_t clockid,
                                const struct __timespec64 *abstime)
{
  unsigned int r;

  /* Make sure any passed in clockid and timeout value are valid.  Note that
     the previous implementation assumed that this check *must* not be
     performed if there would in fact be no blocking; however, POSIX only
     requires that "the validity of the abstime parameter need not be checked
     if the lock can be immediately acquired" (i.e., we need not but may check
     it).  */
  if (abstime && __glibc_unlikely (!futex_abstimed_supported_clockid (clockid)
      || ! valid_nanoseconds (abstime->tv_nsec)))
    return EINVAL;

  /* Make sure we are not holding the rwlock as a writer.  This is a deadlock
     situation we recognize and report.  */
  if (__glibc_unlikely (atomic_load_relaxed (&rwlock->__data.__cur_writer)
			== THREAD_GETMEM (THREAD_SELF, tid)))
    return EDEADLK;

  /* If we prefer writers, recursive rdlock is disallowed, we are in a read
     phase, and there are other readers present, we try to wait without
     extending the read phase.  We will be unblocked by either one of the
     other active readers, or if the writer gives up WRLOCKED (e.g., on
     timeout).
     If there are no other readers, we simply race with any existing primary
     writer; it would have been a race anyway, and changing the odds slightly
     will likely not make a big difference.  */
  if (rwlock->__data.__flags == PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP)
    {
      r = atomic_load_relaxed (&rwlock->__data.__readers);
      while ((r & PTHREAD_RWLOCK_WRPHASE) == 0
	     && (r & PTHREAD_RWLOCK_WRLOCKED) != 0
	     && (r >> PTHREAD_RWLOCK_READER_SHIFT) > 0)
	{
	  /* TODO Spin first.  */
	  /* Try setting the flag signaling that we are waiting without having
	     incremented the number of readers.  Relaxed MO is fine because
	     this is just about waiting for a state change in __readers.  */
	  if (atomic_compare_exchange_weak_relaxed
	      (&rwlock->__data.__readers, &r, r | PTHREAD_RWLOCK_RWAITING))
	    {
	      /* Wait for as long as the flag is set.  An ABA situation is
		 harmless because the flag is just about the state of
		 __readers, and all threads set the flag under the same
		 conditions.  */
	      while (((r = atomic_load_relaxed (&rwlock->__data.__readers))
		      & PTHREAD_RWLOCK_RWAITING) != 0)
		{
		  int private = __pthread_rwlock_get_private (rwlock);
		  int err = __futex_abstimed_wait64 (&rwlock->__data.__readers,
		                                     r, clockid, abstime,
		                                     private);
		  /* We ignore EAGAIN and EINTR.  On time-outs, we can just
		     return because we don't need to clean up anything.  */
		  if (err == ETIMEDOUT || err == EOVERFLOW)
		    return err;
		}
	      /* It makes sense to not break out of the outer loop here
		 because we might be in the same situation again.  */
	    }
	  else
	    {
	      /* TODO Back-off.  */
	    }
	}
    }
  /* Register as a reader, using an add-and-fetch so that R can be used as
     expected value for future operations.  Acquire MO so we synchronize with
     prior writers as well as the last reader of the previous read phase (see
     below).  */
  r = (atomic_fetch_add_acquire (&rwlock->__data.__readers,
				 (1 << PTHREAD_RWLOCK_READER_SHIFT))
       + (1 << PTHREAD_RWLOCK_READER_SHIFT));

  /* Check whether there is an overflow in the number of readers.  We assume
     that the total number of threads is less than half the maximum number
     of readers that we have bits for in __readers (i.e., with 32-bit int and
     PTHREAD_RWLOCK_READER_SHIFT of 3, we assume there are less than
     1 << (32-3-1) concurrent threads).
     If there is an overflow, we use a CAS to try to decrement the number of
     readers if there still is an overflow situation.  If so, we return
     EAGAIN; if not, we are not a thread causing an overflow situation, and so
     we just continue.  Using a fetch-add instead of the CAS isn't possible
     because other readers might release the lock concurrently, which could
     make us the last reader and thus responsible for handing ownership over
     to writers (which requires a CAS too to make the decrement and ownership
     transfer indivisible).  */
  while (__glibc_unlikely (r >= PTHREAD_RWLOCK_READER_OVERFLOW))
    {
      /* Relaxed MO is okay because we just want to undo our registration and
	 cannot have changed the rwlock state substantially if the CAS
	 succeeds.  */
      if (atomic_compare_exchange_weak_relaxed
	  (&rwlock->__data.__readers,
	   &r, r - (1 << PTHREAD_RWLOCK_READER_SHIFT)))
	return EAGAIN;
    }

  /* We have registered as a reader, so if we are in a read phase, we have
     acquired a read lock.  This is also the reader--reader fast-path.
     Even if there is a primary writer, we just return.  If writers are to
     be preferred and we are the only active reader, we could try to enter a
     write phase to let the writer proceed.  This would be okay because we
     cannot have acquired the lock previously as a reader (which could result
     in deadlock if we would wait for the primary writer to run).  However,
     this seems to be a corner case and handling it specially not be worth the
     complexity.  */
  if (__glibc_likely ((r & PTHREAD_RWLOCK_WRPHASE) == 0))
    return 0;
  /* Otherwise, if we were in a write phase (states #6 or #8), we must wait
     for explicit hand-over of the read phase; the only exception is if we
     can start a read phase if there is no primary writer currently.  */
  while ((r & PTHREAD_RWLOCK_WRPHASE) != 0
	 && (r & PTHREAD_RWLOCK_WRLOCKED) == 0)
    {
      /* Try to enter a read phase: If the CAS below succeeds, we have
	 ownership; if it fails, we will simply retry and reassess the
	 situation.
	 Acquire MO so we synchronize with prior writers.  */
      if (atomic_compare_exchange_weak_acquire (&rwlock->__data.__readers, &r,
						r ^ PTHREAD_RWLOCK_WRPHASE))
	{
	  /* We started the read phase, so we are also responsible for
	     updating the write-phase futex.  Relaxed MO is sufficient.
	     We have to do the same steps as a writer would when handing
	     over the read phase to us because other readers cannot
	     distinguish between us and the writer; this includes
	     explicit hand-over and potentially having to wake other readers
	     (but we can pretend to do the setting and unsetting of WRLOCKED
	     atomically, and thus can skip this step).  */
	  if ((atomic_exchange_relaxed (&rwlock->__data.__wrphase_futex, 0)
	       & PTHREAD_RWLOCK_FUTEX_USED) != 0)
	    {
	      int private = __pthread_rwlock_get_private (rwlock);
	      futex_wake (&rwlock->__data.__wrphase_futex, INT_MAX, private);
	    }
	  return 0;
	}
      else
	{
	  /* TODO Back off before retrying.  Also see above.  */
	}
    }

  /* We were in a write phase but did not install the read phase.  We cannot
     distinguish between a writer and another reader starting the read phase,
     so we must wait for explicit hand-over via __wrphase_futex.
     However, __wrphase_futex might not have been set to 1 yet (either
     because explicit hand-over to the writer is still ongoing, or because
     the writer has started the write phase but has not yet updated
     __wrphase_futex).  The least recent value of __wrphase_futex we can
     read from here is the modification of the last read phase (because
     we synchronize with the last reader in this read phase through
     __readers; see the use of acquire MO on the fetch_add above).
     Therefore, if we observe a value of 0 for __wrphase_futex, we need
     to subsequently check that __readers now indicates a read phase; we
     need to use acquire MO for this so that if we observe a read phase,
     we will also see the modification of __wrphase_futex by the previous
     writer.  We then need to load __wrphase_futex again and continue to
     wait if it is not 0, so that we do not skip explicit hand-over.
     Relaxed MO is sufficient for the load from __wrphase_futex because
     we just use it as an indicator for when we can proceed; we use
     __readers and the acquire MO accesses to it to eventually read from
     the proper stores to __wrphase_futex.  */
  unsigned int wpf;
  bool ready = false;
  for (;;)
    {
      while (((wpf = atomic_load_relaxed (&rwlock->__data.__wrphase_futex))
	      | PTHREAD_RWLOCK_FUTEX_USED) == (1 | PTHREAD_RWLOCK_FUTEX_USED))
	{
	  int private = __pthread_rwlock_get_private (rwlock);
	  if (((wpf & PTHREAD_RWLOCK_FUTEX_USED) == 0)
	      && (!atomic_compare_exchange_weak_relaxed
		  (&rwlock->__data.__wrphase_futex,
		   &wpf, wpf | PTHREAD_RWLOCK_FUTEX_USED)))
	    continue;
	  int err = __futex_abstimed_wait64 (&rwlock->__data.__wrphase_futex,
					     1 | PTHREAD_RWLOCK_FUTEX_USED,
					     clockid, abstime, private);
	  if (err == ETIMEDOUT || err == EOVERFLOW)
	    {
	      /* If we timed out, we need to unregister.  If no read phase
		 has been installed while we waited, we can just decrement
		 the number of readers.  Otherwise, we just acquire the
		 lock, which is allowed because we give no precise timing
		 guarantees, and because the timeout is only required to
		 be in effect if we would have had to wait for other
		 threads (e.g., if futex_wait would time-out immediately
		 because the given absolute time is in the past).  */
	      r = atomic_load_relaxed (&rwlock->__data.__readers);
	      while ((r & PTHREAD_RWLOCK_WRPHASE) != 0)
		{
		  /* We don't need to make anything else visible to
		     others besides unregistering, so relaxed MO is
		     sufficient.  */
		  if (atomic_compare_exchange_weak_relaxed
		      (&rwlock->__data.__readers, &r,
		       r - (1 << PTHREAD_RWLOCK_READER_SHIFT)))
		    return err;
		  /* TODO Back-off.  */
		}
	      /* Use the acquire MO fence to mirror the steps taken in the
		 non-timeout case.  Note that the read can happen both
		 in the atomic_load above as well as in the failure case
		 of the CAS operation.  */
	      atomic_thread_fence_acquire ();
	      /* We still need to wait for explicit hand-over, but we must
		 not use futex_wait anymore because we would just time out
		 in this case and thus make the spin-waiting we need
		 unnecessarily expensive.  */
	      while ((atomic_load_relaxed (&rwlock->__data.__wrphase_futex)
		      | PTHREAD_RWLOCK_FUTEX_USED)
		     == (1 | PTHREAD_RWLOCK_FUTEX_USED))
		{
		  /* TODO Back-off?  */
		}
	      ready = true;
	      break;
	    }
	  /* If we got interrupted (EINTR) or the futex word does not have the
	     expected value (EAGAIN), retry.  */
	}
      if (ready)
	/* See below.  */
	break;
      /* We need acquire MO here so that we synchronize with the lock
	 release of the writer, and so that we observe a recent value of
	 __wrphase_futex (see below).  */
      if ((atomic_load_acquire (&rwlock->__data.__readers)
	   & PTHREAD_RWLOCK_WRPHASE) == 0)
	/* We are in a read phase now, so the least recent modification of
	   __wrphase_futex we can read from is the store by the writer
	   with value 1.  Thus, only now we can assume that if we observe
	   a value of 0, explicit hand-over is finished. Retry the loop
	   above one more time.  */
	ready = true;
    }

  return 0;
}


static __always_inline void
__pthread_rwlock_wrunlock (pthread_rwlock_t *rwlock)
{
  int private = __pthread_rwlock_get_private (rwlock);

  atomic_store_relaxed (&rwlock->__data.__cur_writer, 0);
  /* Disable waiting by writers.  We will wake up after we decided how to
     proceed.  */
  bool wake_writers
    = ((atomic_exchange_relaxed (&rwlock->__data.__writers_futex, 0)
	& PTHREAD_RWLOCK_FUTEX_USED) != 0);

  if (rwlock->__data.__flags != PTHREAD_RWLOCK_PREFER_READER_NP)
    {
      /* First, try to hand over to another writer.  */
      unsigned int w = atomic_load_relaxed (&rwlock->__data.__writers);
      while (w != 0)
	{
	  /* Release MO so that another writer that gets WRLOCKED from us will
	     synchronize with us and thus can take over our view of
	     __readers (including, for example, whether we are in a write
	     phase or not).  */
	  if (atomic_compare_exchange_weak_release
	      (&rwlock->__data.__writers, &w, w | PTHREAD_RWLOCK_WRHANDOVER))
	    /* Another writer will take over.  */
	    goto done;
	  /* TODO Back-off.  */
	}
    }

  /* We have done everything we needed to do to prefer writers, so now we
     either hand over explicitly to readers if there are any, or we simply
     stay in a write phase.  See pthread_rwlock_rdunlock for more details.  */
  unsigned int r = atomic_load_relaxed (&rwlock->__data.__readers);
  /* Release MO so that subsequent readers or writers synchronize with us.  */
  while (!atomic_compare_exchange_weak_release
	 (&rwlock->__data.__readers, &r,
	  ((r ^ PTHREAD_RWLOCK_WRLOCKED)
	   ^ ((r >> PTHREAD_RWLOCK_READER_SHIFT) == 0 ? 0
	      : PTHREAD_RWLOCK_WRPHASE))))
    {
      /* TODO Back-off.  */
    }
  if ((r >> PTHREAD_RWLOCK_READER_SHIFT) != 0)
    {
      /* We must hand over explicitly through __wrphase_futex.  Relaxed MO is
	 sufficient because it is just used to delay acquisition by a writer;
	 any other synchronizes-with relations that are necessary are
	 established through __readers.  */
      if ((atomic_exchange_relaxed (&rwlock->__data.__wrphase_futex, 0)
	   & PTHREAD_RWLOCK_FUTEX_USED) != 0)
	futex_wake (&rwlock->__data.__wrphase_futex, INT_MAX, private);
    }

 done:
  /* We released WRLOCKED in some way, so wake a writer.  */
  if (wake_writers)
    futex_wake (&rwlock->__data.__writers_futex, 1, private);
}


static __always_inline int
__pthread_rwlock_wrlock_full64 (pthread_rwlock_t *rwlock, clockid_t clockid,
                                const struct __timespec64 *abstime)
{
  /* Make sure any passed in clockid and timeout value are valid.  Note that
     the previous implementation assumed that this check *must* not be
     performed if there would in fact be no blocking; however, POSIX only
     requires that "the validity of the abstime parameter need not be checked
     if the lock can be immediately acquired" (i.e., we need not but may check
     it).  */
  if (abstime && __glibc_unlikely (!futex_abstimed_supported_clockid (clockid)
      || ! valid_nanoseconds (abstime->tv_nsec)))
    return EINVAL;

  /* Make sure we are not holding the rwlock as a writer.  This is a deadlock
     situation we recognize and report.  */
  if (__glibc_unlikely (atomic_load_relaxed (&rwlock->__data.__cur_writer)
			== THREAD_GETMEM (THREAD_SELF, tid)))
    return EDEADLK;

  /* First we try to acquire the role of primary writer by setting WRLOCKED;
     if it was set before, there already is a primary writer.  Acquire MO so
     that we synchronize with previous primary writers.

     We do not try to change to a write phase right away using a fetch_or
     because we would have to reset it again and wake readers if there are
     readers present (some readers could try to acquire the lock more than
     once, so setting a write phase in the middle of this could cause
     deadlock).  Changing to a write phase eagerly would only speed up the
     transition from a read phase to a write phase in the uncontended case,
     but it would slow down the contended case if readers are preferred (which
     is the default).
     We could try to CAS from a state with no readers to a write phase, but
     this could be less scalable if readers arrive and leave frequently.  */
  bool may_share_futex_used_flag = false;
  unsigned int r = atomic_fetch_or_acquire (&rwlock->__data.__readers,
					    PTHREAD_RWLOCK_WRLOCKED);
  if (__glibc_unlikely ((r & PTHREAD_RWLOCK_WRLOCKED) != 0))
    {
      /* There is another primary writer.  */
      bool prefer_writer
	= (rwlock->__data.__flags != PTHREAD_RWLOCK_PREFER_READER_NP);
      if (prefer_writer)
	{
	  /* We register as a waiting writer, so that we can make use of
	     writer--writer hand-over.  Relaxed MO is fine because we just
	     want to register.  We assume that the maximum number of threads
	     is less than the capacity in __writers.  */
	  atomic_fetch_add_relaxed (&rwlock->__data.__writers, 1);
	}
      for (;;)
	{
	  /* TODO Spin until WRLOCKED is 0 before trying the CAS below.
	     But pay attention to not delay trying writer--writer hand-over
	     for too long (which we must try eventually anyway).  */
	  if ((r & PTHREAD_RWLOCK_WRLOCKED) == 0)
	    {
	      /* Try to become the primary writer or retry.  Acquire MO as in
		 the fetch_or above.  */
	      if (atomic_compare_exchange_weak_acquire
		  (&rwlock->__data.__readers, &r, r | PTHREAD_RWLOCK_WRLOCKED))
		{
		  if (prefer_writer)
		    {
		      /* Unregister as a waiting writer.  Note that because we
			 acquired WRLOCKED, WRHANDOVER will not be set.
			 Acquire MO on the CAS above ensures that
			 unregistering happens after the previous writer;
			 this sorts the accesses to __writers by all
			 primary writers in a useful way (e.g., any other
			 primary writer acquiring after us or getting it from
			 us through WRHANDOVER will see both our changes to
			 __writers).
			 ??? Perhaps this is not strictly necessary for
			 reasons we do not yet know of.  */
		      atomic_fetch_add_relaxed (&rwlock->__data.__writers, -1);
		    }
		  break;
		}
	      /* Retry if the CAS fails (r will have been updated).  */
	      continue;
	    }
	  /* If writer--writer hand-over is available, try to become the
	     primary writer this way by grabbing the WRHANDOVER token.  If we
	     succeed, we own WRLOCKED.  */
	  if (prefer_writer)
	    {
	      unsigned int w = atomic_load_relaxed (&rwlock->__data.__writers);
	      if ((w & PTHREAD_RWLOCK_WRHANDOVER) != 0)
		{
		  /* Acquire MO is required here so that we synchronize with
		     the writer that handed over WRLOCKED.  We also need this
		     for the reload of __readers below because our view of
		     __readers must be at least as recent as the view of the
		     writer that handed over WRLOCKED; we must avoid an ABA
		     through WRHANDOVER, which could, for example, lead to us
		     assuming we are still in a write phase when in fact we
		     are not.  */
		  if (atomic_compare_exchange_weak_acquire
		      (&rwlock->__data.__writers,
		       &w, (w - PTHREAD_RWLOCK_WRHANDOVER - 1)))
		    {
		      /* Reload so our view is consistent with the view of
			 the previous owner of WRLOCKED.  See above.  */
		      r = atomic_load_relaxed (&rwlock->__data.__readers);
		      break;
		    }
		  /* We do not need to reload __readers here.  We should try
		     to perform writer--writer hand-over if possible; if it
		     is not possible anymore, we will reload __readers
		     elsewhere in this loop.  */
		  continue;
		}
	    }
	  /* We did not acquire WRLOCKED nor were able to use writer--writer
	     hand-over, so we block on __writers_futex.  */
	  int private = __pthread_rwlock_get_private (rwlock);
	  unsigned int wf
	    = atomic_load_relaxed (&rwlock->__data.__writers_futex);
	  if (((wf & ~(unsigned int) PTHREAD_RWLOCK_FUTEX_USED) != 1)
	      || ((wf != (1 | PTHREAD_RWLOCK_FUTEX_USED))
		  && (!atomic_compare_exchange_weak_relaxed
		      (&rwlock->__data.__writers_futex, &wf,
		       1 | PTHREAD_RWLOCK_FUTEX_USED))))
	    {
	      /* If we cannot block on __writers_futex because there is no
		 primary writer, or we cannot set PTHREAD_RWLOCK_FUTEX_USED,
		 we retry.  We must reload __readers here in case we cannot
		 block on __writers_futex so that we can become the primary
		 writer and are not stuck in a loop that just continuously
		 fails to block on __writers_futex.  */
	      r = atomic_load_relaxed (&rwlock->__data.__readers);
	      continue;
	    }
	  /* We set the flag that signals that the futex is used, or we could
	     have set it if we had been faster than other waiters.  As a
	     result, we may share the flag with an unknown number of other
	     writers.  Therefore, we must keep this flag set when we acquire
	     the lock.  We do not need to do this when we do not reach this
	     point here because then we are not part of the group that may
	     share the flag, and another writer will wake one of the writers
	     in this group.  */
	  may_share_futex_used_flag = true;
	  int err = __futex_abstimed_wait64 (&rwlock->__data.__writers_futex,
					     1 | PTHREAD_RWLOCK_FUTEX_USED,
					     clockid, abstime, private);
	  if (err == ETIMEDOUT || err == EOVERFLOW)
	    {
	      if (prefer_writer)
		{
		  /* We need to unregister as a waiting writer.  If we are the
		     last writer and writer--writer hand-over is available,
		     we must make use of it because nobody else will reset
		     WRLOCKED otherwise.  (If we use it, we simply pretend
		     that this happened before the timeout; see
		     pthread_rwlock_rdlock_full for the full reasoning.)
		     Also see the similar code above.  */
		  unsigned int w
		    = atomic_load_relaxed (&rwlock->__data.__writers);
		  while (!atomic_compare_exchange_weak_acquire
			 (&rwlock->__data.__writers, &w,
			  (w == PTHREAD_RWLOCK_WRHANDOVER + 1 ? 0 : w - 1)))
		    {
		      /* TODO Back-off.  */
		    }
		  if (w == PTHREAD_RWLOCK_WRHANDOVER + 1)
		    {
		      /* We must continue as primary writer.  See above.  */
		      r = atomic_load_relaxed (&rwlock->__data.__readers);
		      break;
		    }
		}
	      /* We cleaned up and cannot have stolen another waiting writer's
		 futex wake-up, so just return.  */
	      return err;
	    }
	  /* If we got interrupted (EINTR) or the futex word does not have the
	     expected value (EAGAIN), retry after reloading __readers.  */
	  r = atomic_load_relaxed (&rwlock->__data.__readers);
	}
      /* Our snapshot of __readers is up-to-date at this point because we
	 either set WRLOCKED using a CAS (and update r accordingly below,
	 which was used as expected value for the CAS) or got WRLOCKED from
	 another writer whose snapshot of __readers we inherit.  */
      r |= PTHREAD_RWLOCK_WRLOCKED;
    }

  /* We are the primary writer; enable blocking on __writers_futex.  Relaxed
     MO is sufficient for futex words; acquire MO on the previous
     modifications of __readers ensures that this store happens after the
     store of value 0 by the previous primary writer.  */
  atomic_store_relaxed (&rwlock->__data.__writers_futex,
			1 | (may_share_futex_used_flag
			     ? PTHREAD_RWLOCK_FUTEX_USED : 0));

  /* If we are in a write phase, we have acquired the lock.  */
  if ((r & PTHREAD_RWLOCK_WRPHASE) != 0)
    goto done;

  /* If we are in a read phase and there are no readers, try to start a write
     phase.  */
  while ((r & PTHREAD_RWLOCK_WRPHASE) == 0
	 && (r >> PTHREAD_RWLOCK_READER_SHIFT) == 0)
    {
      /* Acquire MO so that we synchronize with prior writers and do
	 not interfere with their updates to __writers_futex, as well
	 as regarding prior readers and their updates to __wrphase_futex,
	 respectively.  */
      if (atomic_compare_exchange_weak_acquire (&rwlock->__data.__readers,
						&r, r | PTHREAD_RWLOCK_WRPHASE))
	{
	  /* We have started a write phase, so need to enable readers to wait.
	     See the similar case in __pthread_rwlock_rdlock_full.  Unlike in
	     that similar case, we are the (only) primary writer and so do
	     not need to wake another writer.  */
	  atomic_store_relaxed (&rwlock->__data.__wrphase_futex, 1);

	  goto done;
	}
      /* TODO Back-off.  */
    }

  /* We became the primary writer in a read phase and there were readers when
     we did (because of the previous loop).  Thus, we have to wait for
     explicit hand-over from one of these readers.
     We basically do the same steps as for the similar case in
     __pthread_rwlock_rdlock_full, except that we additionally might try
     to directly hand over to another writer and need to wake up
     other writers or waiting readers (i.e., PTHREAD_RWLOCK_RWAITING).  */
  unsigned int wpf;
  bool ready = false;
  for (;;)
    {
      while (((wpf = atomic_load_relaxed (&rwlock->__data.__wrphase_futex))
	      | PTHREAD_RWLOCK_FUTEX_USED) == PTHREAD_RWLOCK_FUTEX_USED)
	{
	  int private = __pthread_rwlock_get_private (rwlock);
	  if ((wpf & PTHREAD_RWLOCK_FUTEX_USED) == 0
	      && (!atomic_compare_exchange_weak_relaxed
		  (&rwlock->__data.__wrphase_futex, &wpf,
		   PTHREAD_RWLOCK_FUTEX_USED)))
	    continue;
	  int err = __futex_abstimed_wait64 (&rwlock->__data.__wrphase_futex,
					     PTHREAD_RWLOCK_FUTEX_USED,
					     clockid, abstime, private);
	  if (err == ETIMEDOUT || err == EOVERFLOW)
	    {
	      if (rwlock->__data.__flags != PTHREAD_RWLOCK_PREFER_READER_NP)
		{
		  /* We try writer--writer hand-over.  */
		  unsigned int w
		    = atomic_load_relaxed (&rwlock->__data.__writers);
		  if (w != 0)
		    {
		      /* We are about to hand over WRLOCKED, so we must
			 release __writers_futex too; otherwise, we'd have
			 a pending store, which could at least prevent
			 other threads from waiting using the futex
			 because it could interleave with the stores
			 by subsequent writers.  In turn, this means that
			 we have to clean up when we do not hand over
			 WRLOCKED.
			 Release MO so that another writer that gets
			 WRLOCKED from us can take over our view of
			 __readers.  */
		      unsigned int wf
			= atomic_exchange_relaxed (&rwlock->__data.__writers_futex, 0);
		      while (w != 0)
			{
			  if (atomic_compare_exchange_weak_release
			      (&rwlock->__data.__writers, &w,
			       w | PTHREAD_RWLOCK_WRHANDOVER))
			    {
			      /* Wake other writers.  */
			      if ((wf & PTHREAD_RWLOCK_FUTEX_USED) != 0)
				futex_wake (&rwlock->__data.__writers_futex,
					    1, private);
			      return err;
			    }
			  /* TODO Back-off.  */
			}
		      /* We still own WRLOCKED and someone else might set
			 a write phase concurrently, so enable waiting
			 again.  Make sure we don't loose the flag that
			 signals whether there are threads waiting on
			 this futex.  */
		      atomic_store_relaxed (&rwlock->__data.__writers_futex, wf);
		    }
		}
	      /* If we timed out and we are not in a write phase, we can
		 just stop being a primary writer.  Otherwise, we just
		 acquire the lock.  */
	      r = atomic_load_relaxed (&rwlock->__data.__readers);
	      if ((r & PTHREAD_RWLOCK_WRPHASE) == 0)
		{
		  /* We are about to release WRLOCKED, so we must release
		     __writers_futex too; see the handling of
		     writer--writer hand-over above.  */
		  unsigned int wf
		    = atomic_exchange_relaxed (&rwlock->__data.__writers_futex, 0);
		  while ((r & PTHREAD_RWLOCK_WRPHASE) == 0)
		    {
		      /* While we don't need to make anything from a
			 caller's critical section visible to other
			 threads, we need to ensure that our changes to
			 __writers_futex are properly ordered.
			 Therefore, use release MO to synchronize with
			 subsequent primary writers.  Also wake up any
			 waiting readers as they are waiting because of
			 us.  */
		      if (atomic_compare_exchange_weak_release
			  (&rwlock->__data.__readers, &r,
			   (r ^ PTHREAD_RWLOCK_WRLOCKED)
			   & ~(unsigned int) PTHREAD_RWLOCK_RWAITING))
			{
			  /* Wake other writers.  */
			  if ((wf & PTHREAD_RWLOCK_FUTEX_USED) != 0)
			    futex_wake (&rwlock->__data.__writers_futex,
					1, private);
			  /* Wake waiting readers.  */
			  if ((r & PTHREAD_RWLOCK_RWAITING) != 0)
			    futex_wake (&rwlock->__data.__readers,
					INT_MAX, private);
			  return ETIMEDOUT;
			}
		    }
		  /* We still own WRLOCKED and someone else might set a
		     write phase concurrently, so enable waiting again.
		     Make sure we don't loose the flag that signals
		     whether there are threads waiting on this futex.  */
		  atomic_store_relaxed (&rwlock->__data.__writers_futex, wf);
		}
	      /* Use the acquire MO fence to mirror the steps taken in the
		 non-timeout case.  Note that the read can happen both
		 in the atomic_load above as well as in the failure case
		 of the CAS operation.  */
	      atomic_thread_fence_acquire ();
	      /* We still need to wait for explicit hand-over, but we must
		 not use futex_wait anymore.  */
	      while ((atomic_load_relaxed (&rwlock->__data.__wrphase_futex)
		      | PTHREAD_RWLOCK_FUTEX_USED)
		     == PTHREAD_RWLOCK_FUTEX_USED)
		{
		  /* TODO Back-off.  */
		}
	      ready = true;
	      break;
	    }
	  /* If we got interrupted (EINTR) or the futex word does not have
	     the expected value (EAGAIN), retry.  */
	}
      /* See pthread_rwlock_rdlock_full.  */
      if (ready)
	break;
      if ((atomic_load_acquire (&rwlock->__data.__readers)
	   & PTHREAD_RWLOCK_WRPHASE) != 0)
	ready = true;
    }

 done:
  atomic_store_relaxed (&rwlock->__data.__cur_writer,
			THREAD_GETMEM (THREAD_SELF, tid));
  return 0;
}
