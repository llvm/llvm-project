/* elision-lock.c: Elided pthread mutex lock.
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

#include <stdio.h>
#include <pthread.h>
#include <pthreadP.h>
#include <lowlevellock.h>
#include <elision-conf.h>
#include "htm.h"

#ifndef EXTRAARG
# define EXTRAARG
#endif
#ifndef LLL_LOCK
# define LLL_LOCK(a,b) lll_lock(a,b), 0
#endif

#define aconf __elision_aconf

/* Adaptive lock using transactions.
   By default the lock region is run as a transaction, and when it
   aborts or the lock is busy the lock adapts itself.  */

int
__lll_lock_elision (int *lock, short *adapt_count, EXTRAARG int pshared)
{
  /* adapt_count is accessed concurrently but is just a hint.  Thus,
     use atomic accesses but relaxed MO is sufficient.  */
  if (atomic_load_relaxed (adapt_count) > 0)
    {
      goto use_lock;
    }

  for (int i = aconf.try_tbegin; i > 0; i--)
    {
      if (__libc_tbegin (0))
	{
	  if (*lock == 0)
	    return 0;
	  /* Lock was busy.  Fall back to normal locking.  */
	  __libc_tabort (_ABORT_LOCK_BUSY);
	}
      else
	{
	  /* A persistent failure indicates that a retry will probably
	     result in another failure.  Use normal locking now and
	     for the next couple of calls.  */
	  if (_TEXASRU_FAILURE_PERSISTENT (__builtin_get_texasru ()))
	    {
	      if (aconf.skip_lock_internal_abort > 0)
		atomic_store_relaxed (adapt_count,
				      aconf.skip_lock_internal_abort);
	      goto use_lock;
	    }
	}
     }

  /* Fall back to locks for a bit if retries have been exhausted */
  if (aconf.try_tbegin > 0 && aconf.skip_lock_out_of_tbegin_retries > 0)
    atomic_store_relaxed (adapt_count,
			  aconf.skip_lock_out_of_tbegin_retries);

use_lock:
  return LLL_LOCK ((*lock), pshared);
}
libc_hidden_def (__lll_lock_elision)
