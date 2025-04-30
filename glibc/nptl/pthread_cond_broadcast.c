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
#include <stap-probe.h>
#include <atomic.h>

#include <shlib-compat.h>

#include "pthread_cond_common.c"


/* We do the following steps from __pthread_cond_signal in one critical
   section: (1) signal all waiters in G1, (2) close G1 so that it can become
   the new G2 and make G2 the new G1, and (3) signal all waiters in the new
   G1.  We don't need to do all these steps if there are no waiters in G1
   and/or G2.  See __pthread_cond_signal for further details.  */
int
___pthread_cond_broadcast (pthread_cond_t *cond)
{
  LIBC_PROBE (cond_broadcast, 1, cond);

  unsigned int wrefs = atomic_load_relaxed (&cond->__data.__wrefs);
  if (wrefs >> 3 == 0)
    return 0;
  int private = __condvar_get_private (wrefs);

  __condvar_acquire_lock (cond, private);

  unsigned long long int wseq = __condvar_load_wseq_relaxed (cond);
  unsigned int g2 = wseq & 1;
  unsigned int g1 = g2 ^ 1;
  wseq >>= 1;
  bool do_futex_wake = false;

  /* Step (1): signal all waiters remaining in G1.  */
  if (cond->__data.__g_size[g1] != 0)
    {
      /* Add as many signals as the remaining size of the group.  */
      atomic_fetch_add_relaxed (cond->__data.__g_signals + g1,
				cond->__data.__g_size[g1] << 1);
      cond->__data.__g_size[g1] = 0;

      /* We need to wake G1 waiters before we quiesce G1 below.  */
      /* TODO Only set it if there are indeed futex waiters.  We could
	 also try to move this out of the critical section in cases when
	 G2 is empty (and we don't need to quiesce).  */
      futex_wake (cond->__data.__g_signals + g1, INT_MAX, private);
    }

  /* G1 is complete.  Step (2) is next unless there are no waiters in G2, in
     which case we can stop.  */
  if (__condvar_quiesce_and_switch_g1 (cond, wseq, &g1, private))
    {
      /* Step (3): Send signals to all waiters in the old G2 / new G1.  */
      atomic_fetch_add_relaxed (cond->__data.__g_signals + g1,
				cond->__data.__g_size[g1] << 1);
      cond->__data.__g_size[g1] = 0;
      /* TODO Only set it if there are indeed futex waiters.  */
      do_futex_wake = true;
    }

  __condvar_release_lock (cond, private);

  if (do_futex_wake)
    futex_wake (cond->__data.__g_signals + g1, INT_MAX, private);

  return 0;
}
versioned_symbol (libc, ___pthread_cond_broadcast,
		  pthread_cond_broadcast, GLIBC_2_3_2);
libc_hidden_ver (___pthread_cond_broadcast, __pthread_cond_broadcast)
#ifndef SHARED
strong_alias (___pthread_cond_broadcast, __pthread_cond_broadcast)
#endif
