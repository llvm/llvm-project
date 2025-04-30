/* Bug 23844: Test for pthread_rwlock_tryrdlock stalls.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* For a full analysis see comment:
   https://sourceware.org/bugzilla/show_bug.cgi?id=23844#c14

   Provided here for reference:

   --- Analysis of pthread_rwlock_tryrdlock() stall ---
   A read lock begins to execute.

   In __pthread_rwlock_rdlock_full:

   We can attempt a read lock, but find that the lock is
   in a write phase (PTHREAD_RWLOCK_WRPHASE, or WP-bit
   is set), and the lock is held by a primary writer
   (PTHREAD_RWLOCK_WRLOCKED is set). In this case we must
   wait for explicit hand over from the writer to us or
   one of the other waiters. The read lock threads are
   about to execute:

   341   r = (atomic_fetch_add_acquire (&rwlock->__data.__readers,
   342                                  (1 << PTHREAD_RWLOCK_READER_SHIFT))
   343        + (1 << PTHREAD_RWLOCK_READER_SHIFT));

   An unlock beings to execute.

   Then in __pthread_rwlock_wrunlock:

   547   unsigned int r = atomic_load_relaxed (&rwlock->__data.__readers);
   ...
   549   while (!atomic_compare_exchange_weak_release
   550          (&rwlock->__data.__readers, &r,
   551           ((r ^ PTHREAD_RWLOCK_WRLOCKED)
   552            ^ ((r >> PTHREAD_RWLOCK_READER_SHIFT) == 0 ? 0
   553               : PTHREAD_RWLOCK_WRPHASE))))
   554     {
   ...
   556     }

   We clear PTHREAD_RWLOCK_WRLOCKED, and if there are
   no readers so we leave the lock in PTHRAD_RWLOCK_WRPHASE.

   Back in the read lock.

   The read lock adjusts __readres as above.

   383   while ((r & PTHREAD_RWLOCK_WRPHASE) != 0
   384          && (r & PTHREAD_RWLOCK_WRLOCKED) == 0)
   385     {
   ...
   390       if (atomic_compare_exchange_weak_acquire (&rwlock->__data.__readers, &r,
   391                                                 r ^ PTHREAD_RWLOCK_WRPHASE))
   392         {

   And then attemps to start the read phase.

   Assume there happens to be a tryrdlock at this point, noting
   that PTHREAD_RWLOCK_WRLOCKED is clear, and PTHREAD_RWLOCK_WRPHASE
   is 1. So the try lock attemps to start the read phase.

   In __pthread_rwlock_tryrdlock:

    44       if ((r & PTHREAD_RWLOCK_WRPHASE) == 0)
    45         {
   ...
    49           if (((r & PTHREAD_RWLOCK_WRLOCKED) != 0)
    50               && (rwlock->__data.__flags
    51                   == PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP))
    52             return EBUSY;
    53           rnew = r + (1 << PTHREAD_RWLOCK_READER_SHIFT);
    54         }
   ...
    89   while (!atomic_compare_exchange_weak_acquire (&rwlock->__data.__readers,
    90       &r, rnew));

   And succeeds.

   Back in the write unlock:

   557   if ((r >> PTHREAD_RWLOCK_READER_SHIFT) != 0)
   558     {
   ...
   563       if ((atomic_exchange_relaxed (&rwlock->__data.__wrphase_futex, 0)
   564            & PTHREAD_RWLOCK_FUTEX_USED) != 0)
   565         futex_wake (&rwlock->__data.__wrphase_futex, INT_MAX, private);
   566     }

   We note that PTHREAD_RWLOCK_FUTEX_USED is non-zero
   and don't wake anyone. This is OK because we handed
   over to the trylock. It will be the trylock's responsibility
   to wake any waiters.

   Back in the read lock:

   The read lock fails to install PTHRAD_REWLOCK_WRPHASE as 0 because
   the __readers value was adjusted by the trylock, and so it falls through
   to waiting on the lock for explicit handover from either a new writer
   or a new reader.

   448           int err = futex_abstimed_wait (&rwlock->__data.__wrphase_futex,
   449                                          1 | PTHREAD_RWLOCK_FUTEX_USED,
   450                                          abstime, private);

   We use PTHREAD_RWLOCK_FUTEX_USED to indicate the futex
   is in use.

   At this point we have readers waiting on the read lock
   to unlock. The wrlock is done. The trylock is finishing
   the installation of the read phase.

    92   if ((r & PTHREAD_RWLOCK_WRPHASE) != 0)
    93     {
   ...
   105       atomic_store_relaxed (&rwlock->__data.__wrphase_futex, 0);
   106     }

   The trylock does note that we were the one that
   installed the read phase, but the comments are not
   correct, the execution ordering above shows that
   readers might indeed be waiting, and they are.

   The atomic_store_relaxed throws away PTHREAD_RWLOCK_FUTEX_USED,
   and the waiting reader is never worken becuase as noted
   above it is conditional on the futex being used.

   The solution is for the trylock thread to inspect
   PTHREAD_RWLOCK_FUTEX_USED and wake the waiting readers.

   --- Analysis of pthread_rwlock_trywrlock() stall ---

   A write lock begins to execute, takes the write lock,
   and then releases the lock...

   In pthread_rwlock_wrunlock():

   547   unsigned int r = atomic_load_relaxed (&rwlock->__data.__readers);
   ...
   549   while (!atomic_compare_exchange_weak_release
   550          (&rwlock->__data.__readers, &r,
   551           ((r ^ PTHREAD_RWLOCK_WRLOCKED)
   552            ^ ((r >> PTHREAD_RWLOCK_READER_SHIFT) == 0 ? 0
   553               : PTHREAD_RWLOCK_WRPHASE))))
   554     {
   ...
   556     }

   ... leaving it in the write phase with zero readers
   (the case where we leave the write phase in place
   during a write unlock).

   A write trylock begins to execute.

   In __pthread_rwlock_trywrlock:

    40   while (((r & PTHREAD_RWLOCK_WRLOCKED) == 0)
    41       && (((r >> PTHREAD_RWLOCK_READER_SHIFT) == 0)
    42           || (prefer_writer && ((r & PTHREAD_RWLOCK_WRPHASE) != 0))))
    43     {

   The lock is not locked.

   There are no readers.

    45       if (atomic_compare_exchange_weak_acquire (
    46           &rwlock->__data.__readers, &r,
    47           r | PTHREAD_RWLOCK_WRPHASE | PTHREAD_RWLOCK_WRLOCKED))

   We atomically install the write phase and we take the
   exclusive write lock.

    48         {
    49           atomic_store_relaxed (&rwlock->__data.__writers_futex, 1);

   We get this far.

   A reader lock begins to execute.

   In pthread_rwlock_rdlock:

   437   for (;;)
   438     {
   439       while (((wpf = atomic_load_relaxed (&rwlock->__data.__wrphase_futex))
   440               | PTHREAD_RWLOCK_FUTEX_USED) == (1 | PTHREAD_RWLOCK_FUTEX_USED))
   441         {
   442           int private = __pthread_rwlock_get_private (rwlock);
   443           if (((wpf & PTHREAD_RWLOCK_FUTEX_USED) == 0)
   444               && (!atomic_compare_exchange_weak_relaxed
   445                   (&rwlock->__data.__wrphase_futex,
   446                    &wpf, wpf | PTHREAD_RWLOCK_FUTEX_USED)))
   447             continue;
   448           int err = futex_abstimed_wait (&rwlock->__data.__wrphase_futex,
   449                                          1 | PTHREAD_RWLOCK_FUTEX_USED,
   450                                          abstime, private);

   We are in a write phase, so the while() on line 439 is true.

   The value of wpf does not have PTHREAD_RWLOCK_FUTEX_USED set
   since this is the first reader to lock.

   The atomic operation sets wpf with PTHREAD_RELOCK_FUTEX_USED
   on the expectation that this reader will be woken during
   the handoff.

   Back in pthread_rwlock_trywrlock:

    50           atomic_store_relaxed (&rwlock->__data.__wrphase_futex, 1);
    51           atomic_store_relaxed (&rwlock->__data.__cur_writer,
    52               THREAD_GETMEM (THREAD_SELF, tid));
    53           return 0;
    54         }
   ...
    57     }

   We write 1 to __wrphase_futex discarding PTHREAD_RWLOCK_FUTEX_USED,
   and so in the unlock we will not awaken the waiting reader.

   The solution to this is to realize that if we did not start the write
   phase we need not write 1 or any other value to __wrphase_futex.
   This ensures that any readers (which saw __wrphase_futex != 0) can
   set PTHREAD_RWLOCK_FUTEX_USED and this can be used at unlock to
   wake them.

   If we installed the write phase then all other readers are looping
   here:

   In __pthread_rwlock_rdlock_full:

   437   for (;;)
   438     {
   439       while (((wpf = atomic_load_relaxed (&rwlock->__data.__wrphase_futex))
   440               | PTHREAD_RWLOCK_FUTEX_USED) == (1 | PTHREAD_RWLOCK_FUTEX_USED))
   441         {
   ...
   508     }

   waiting for the write phase to be installed or removed before they
   can begin waiting on __wrphase_futex (part of the algorithm), or
   taking a concurrent read lock, and thus we can safely write 1 to
   __wrphase_futex.

   If we did not install the write phase then the readers may already
   be waiting on the futex, the original writer wrote 1 to __wrphase_futex
   as part of starting the write phase, and we cannot also write 1
   without loosing the PTHREAD_RWLOCK_FUTEX_USED bit.

   ---

   Summary for the pthread_rwlock_tryrdlock() stall:

   The stall is caused by pthread_rwlock_tryrdlock failing to check
   that PTHREAD_RWLOCK_FUTEX_USED is set in the __wrphase_futex futex
   and then waking the futex.

   The fix for bug 23844 ensures that waiters on __wrphase_futex are
   correctly woken.  Before the fix the test stalls as readers can
   wait forever on __wrphase_futex.  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <support/xthread.h>
#include <errno.h>

/* We need only one lock to reproduce the issue. We will need multiple
   threads to get the exact case where we have a read, try, and unlock
   all interleaving to produce the case where the readers are waiting
   and the try fails to wake them.  */
pthread_rwlock_t onelock;

/* The number of threads is arbitrary but empirically chosen to have
   enough threads that we see the condition where waiting readers are
   not woken by a successful tryrdlock.  */
#define NTHREADS 32

_Atomic int do_exit;

void *
run_loop (void *arg)
{
  int i = 0, ret;
  while (!do_exit)
    {
      /* Arbitrarily choose if we are the writer or reader.  Choose a
	 high enough ratio of readers to writers to make it likely
	 that readers block (and eventually are susceptable to
	 stalling).

         If we are a writer, take the write lock, and then unlock.
	 If we are a reader, try the lock, then lock, then unlock.  */
      if ((i % 8) != 0)
	xpthread_rwlock_wrlock (&onelock);
      else
	{
	  if ((ret = pthread_rwlock_tryrdlock (&onelock)) != 0)
	    {
	      if (ret == EBUSY)
		xpthread_rwlock_rdlock (&onelock);
	      else
		exit (EXIT_FAILURE);
	    }
	}
      /* Thread does some work and then unlocks.  */
      xpthread_rwlock_unlock (&onelock);
      i++;
    }
  return NULL;
}

int
do_test (void)
{
  int i;
  pthread_t tids[NTHREADS];
  xpthread_rwlock_init (&onelock, NULL);
  for (i = 0; i < NTHREADS; i++)
    tids[i] = xpthread_create (NULL, run_loop, NULL);
  /* Run for some amount of time.  Empirically speaking exercising
     the stall via pthread_rwlock_tryrdlock is much harder, and on
     a 3.5GHz 4 core x86_64 VM system it takes somewhere around
     20-200s to stall, approaching 100% stall past 200s.  We can't
     wait that long for a regression test so we just test for 20s,
     and expect the stall to happen with a 5-10% chance (enough for
     developers to see).  */
  sleep (20);
  /* Then exit.  */
  printf ("INFO: Exiting...\n");
  do_exit = 1;
  /* If any readers stalled then we will timeout waiting for them.  */
  for (i = 0; i < NTHREADS; i++)
    xpthread_join (tids[i]);
  printf ("INFO: Done.\n");
  xpthread_rwlock_destroy (&onelock);
  printf ("PASS: No pthread_rwlock_tryrdlock stalls detected.\n");
  return 0;
}

#define TIMEOUT 30
#include <support/test-driver.c>
