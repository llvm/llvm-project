/* elision-trylock.c: Lock eliding trylock for pthreads.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <elision-conf.h>
#include "htm.h"

#define aconf __elision_aconf

/* Try to elide a futex trylock.  FUTEX is the futex variable.  ADAPT_COUNT is
   the adaptation counter in the mutex.  */

int
__lll_trylock_elision (int *futex, short *adapt_count)
{
  /* Implement POSIX semantics by forbiding nesting elided trylocks.  */
  __libc_tabort (_ABORT_NESTED_TRYLOCK);

  /* Only try a transaction if it's worth it.  */
  if (atomic_load_relaxed (adapt_count) > 0)
    {
      goto use_lock;
    }

  if (__libc_tbegin (0))
    {
      if (*futex == 0)
	return 0;

      /* Lock was busy.  This is never a nested transaction.
         End it, and set the adapt count.  */
      __libc_tend (0);

      if (aconf.skip_lock_busy > 0)
	atomic_store_relaxed (adapt_count, aconf.skip_lock_busy);
    }
  else
    {
      if (_TEXASRU_FAILURE_PERSISTENT (__builtin_get_texasru ()))
	{
	  /* A persistent failure indicates that a retry will probably
	     result in another failure.  Use normal locking now and
	     for the next couple of calls.  */
	  if (aconf.skip_trylock_internal_abort > 0)
	    atomic_store_relaxed (adapt_count,
				aconf.skip_trylock_internal_abort);
	}
    }

use_lock:
  return lll_trylock (*futex);
}
libc_hidden_def (__lll_trylock_elision)
