/* elision-trylock.c: Lock eliding trylock for pthreads.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <pthreadP.h>
#include <lowlevellock.h>
#include "hle.h"
#include <elision-conf.h>

#define aconf __elision_aconf

/* Try to elide a futex trylock.  FUTEX is the futex variable.  ADAPT_COUNT is
   the adaptation counter in the mutex.  */

int
__lll_trylock_elision (int *futex, short *adapt_count)
{
  /* Implement POSIX semantics by forbiding nesting
     trylock.  Sorry.  After the abort the code is re-executed
     non transactional and if the lock was already locked
     return an error.  */
  _xabort (_ABORT_NESTED_TRYLOCK);

  /* Only try a transaction if it's worth it.  See __lll_lock_elision for
     why we need atomic accesses.  Relaxed MO is sufficient because this is
     just a hint.  */
  if (atomic_load_relaxed (adapt_count) <= 0)
    {
      unsigned status;

      if ((status = _xbegin()) == _XBEGIN_STARTED)
	{
	  if (*futex == 0)
	    return 0;

	  /* Lock was busy.  Fall back to normal locking.
	     Could also _xend here but xabort with 0xff code
	     is more visible in the profiler.  */
	  _xabort (_ABORT_LOCK_BUSY);
	}

      if (!(status & _XABORT_RETRY))
        {
          /* Internal abort.  No chance for retry.  For future
             locks don't try speculation for some time.  See above for MO.  */
          if (atomic_load_relaxed (adapt_count)
              != aconf.skip_lock_internal_abort)
            atomic_store_relaxed (adapt_count, aconf.skip_lock_internal_abort);
        }
      /* Could do some retries here.  */
    }
  else
    {
      /* Lost updates are possible but harmless (see above).  */
      atomic_store_relaxed (adapt_count,
	  atomic_load_relaxed (adapt_count) - 1);
    }

  return lll_trylock (*futex);
}
libc_hidden_def (__lll_trylock_elision)
