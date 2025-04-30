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
#include <shlib-compat.h>
#include "pthreadP.h"
#include <stap-probe.h>
#include <atomic.h>
#include <futex-internal.h>

#include "pthread_cond_common.c"

/* See __pthread_cond_wait for a high-level description of the algorithm.

   A correct program must make sure that no waiters are blocked on the condvar
   when it is destroyed, and that there are no concurrent signals or
   broadcasts.  To wake waiters reliably, the program must signal or
   broadcast while holding the mutex or after having held the mutex.  It must
   also ensure that no signal or broadcast are still pending to unblock
   waiters; IOW, because waiters can wake up spuriously, the program must
   effectively ensure that destruction happens after the execution of those
   signal or broadcast calls.
   Thus, we can assume that all waiters that are still accessing the condvar
   have been woken.  We wait until they have confirmed to have woken up by
   decrementing __wrefs.  */
int
__pthread_cond_destroy (pthread_cond_t *cond)
{
  LIBC_PROBE (cond_destroy, 1, cond);

  /* Set the wake request flag.  We could also spin, but destruction that is
     concurrent with still-active waiters is probably neither common nor
     performance critical.  Acquire MO to synchronize with waiters confirming
     that they finished.  */
  unsigned int wrefs = atomic_fetch_or_acquire (&cond->__data.__wrefs, 4);
  int private = __condvar_get_private (wrefs);
  while (wrefs >> 3 != 0)
    {
      futex_wait_simple (&cond->__data.__wrefs, wrefs, private);
      /* See above.  */
      wrefs = atomic_load_acquire (&cond->__data.__wrefs);
    }
  /* The memory the condvar occupies can now be reused.  */
  return 0;
}
libc_hidden_def (__pthread_cond_destroy)
versioned_symbol (libc, __pthread_cond_destroy,
		  pthread_cond_destroy, GLIBC_2_3_2);
