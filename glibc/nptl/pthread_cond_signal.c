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

#include <endian.h>
#include <errno.h>
#include <sysdep.h>
#include <futex-internal.h>
#include <pthread.h>
#include <pthreadP.h>
#include <atomic.h>
#include <stdint.h>

#include <shlib-compat.h>
#include <stap-probe.h>

#include "pthread_cond_common.c"

/* See __pthread_cond_wait for a high-level description of the algorithm.  */
int
___pthread_cond_signal (pthread_cond_t *cond)
{
  LIBC_PROBE (cond_signal, 1, cond);

  /* First check whether there are waiters.  Relaxed MO is fine for that for
     the same reasons that relaxed MO is fine when observing __wseq (see
     below).  */
  unsigned int wrefs = atomic_load_relaxed (&cond->__data.__wrefs);
  if (wrefs >> 3 == 0)
    return 0;
  int private = __condvar_get_private (wrefs);

  __condvar_acquire_lock (cond, private);

  /* Load the waiter sequence number, which represents our relative ordering
     to any waiters.  Relaxed MO is sufficient for that because:
     1) We can pick any position that is allowed by external happens-before
        constraints.  In particular, if another __pthread_cond_wait call
        happened before us, this waiter must be eligible for being woken by
        us.  The only way do establish such a happens-before is by signaling
        while having acquired the mutex associated with the condvar and
        ensuring that the signal's critical section happens after the waiter.
        Thus, the mutex ensures that we see that waiter's __wseq increase.
     2) Once we pick a position, we do not need to communicate this to the
        program via a happens-before that we set up: First, any wake-up could
        be a spurious wake-up, so the program must not interpret a wake-up as
        an indication that the waiter happened before a particular signal;
        second, a program cannot detect whether a waiter has not yet been
        woken (i.e., it cannot distinguish between a non-woken waiter and one
        that has been woken but hasn't resumed execution yet), and thus it
        cannot try to deduce that a signal happened before a particular
        waiter.  */
  unsigned long long int wseq = __condvar_load_wseq_relaxed (cond);
  unsigned int g1 = (wseq & 1) ^ 1;
  wseq >>= 1;
  bool do_futex_wake = false;

  /* If G1 is still receiving signals, we put the signal there.  If not, we
     check if G2 has waiters, and if so, quiesce and switch G1 to the former
     G2; if this results in a new G1 with waiters (G2 might have cancellations
     already, see __condvar_quiesce_and_switch_g1), we put the signal in the
     new G1.  */
  if ((cond->__data.__g_size[g1] != 0)
      || __condvar_quiesce_and_switch_g1 (cond, wseq, &g1, private))
    {
      /* Add a signal.  Relaxed MO is fine because signaling does not need to
	 establish a happens-before relation (see above).  We do not mask the
	 release-MO store when initializing a group in
	 __condvar_quiesce_and_switch_g1 because we use an atomic
	 read-modify-write and thus extend that store's release sequence.  */
      atomic_fetch_add_relaxed (cond->__data.__g_signals + g1, 2);
      cond->__data.__g_size[g1]--;
      /* TODO Only set it if there are indeed futex waiters.  */
      do_futex_wake = true;
    }

  __condvar_release_lock (cond, private);

  if (do_futex_wake)
    futex_wake (cond->__data.__g_signals + g1, 1, private);

  return 0;
}
versioned_symbol (libpthread, ___pthread_cond_signal, pthread_cond_signal,
		  GLIBC_2_3_2);
libc_hidden_ver (___pthread_cond_signal, __pthread_cond_signal)
#ifndef SHARED
strong_alias (___pthread_cond_signal, __pthread_cond_signal)
#endif
