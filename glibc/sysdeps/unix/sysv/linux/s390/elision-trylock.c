/* Elided pthread mutex trylock.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
#include <htm.h>
#include <elision-conf.h>

#define aconf __elision_aconf

/* Try to elide a futex trylock.  FUTEX is the futex variable.  ADAPT_COUNT is
   the adaptation counter in the mutex.  */

int
__lll_trylock_elision (int *futex, short *adapt_count)
{
  /* Implement POSIX semantics by forbiding nesting elided trylocks.
     Sorry.  After the abort the code is re-executed
     non transactional and if the lock was already locked
     return an error.  */
  if (__libc_tx_nesting_depth () > 0)
    {
      /* Note that this abort may terminate an outermost transaction that
	 was created outside glibc.
	 This persistently aborts the current transactions to force
	 them to use the default lock instead of retrying transactions
	 until their try_tbegin is zero.
      */
      __libc_tabort (_HTM_FIRST_USER_ABORT_CODE | 1);
      __builtin_unreachable ();
    }

  /* adapt_count can be accessed concurrently; these accesses can be both
     inside of transactions (if critical sections are nested and the outer
     critical section uses lock elision) and outside of transactions.  Thus,
     we need to use atomic accesses to avoid data races.  However, the
     value of adapt_count is just a hint, so relaxed MO accesses are
     sufficient.  */
    if (atomic_load_relaxed (adapt_count) <= 0 && aconf.try_tbegin > 0)
    {
      int status = __libc_tbegin ((void *) 0);
      if (__glibc_likely (status  == _HTM_TBEGIN_STARTED))
	{
	  /* Check the futex to make sure nobody has touched it in the
	     mean time.  This forces the futex into the cache and makes
	     sure the transaction aborts if another thread acquires the lock
	     concurrently.  */
	  if (__glibc_likely (atomic_load_relaxed (futex) == 0))
	    /* Lock was free.  Return to user code in a transaction.  */
	    return 0;

	  /* Lock was busy.  Fall back to normal locking.
	     This can be the case if e.g. adapt_count was decremented to zero
	     by a former release and another thread has been waken up and
	     acquired it.
	     Since we are in a non-nested transaction there is no need to abort,
	     which is expensive.  Simply end the started transaction.  */
	  __libc_tend ();
	  /* Note: Changing the adapt_count here might abort a transaction on a
	     different CPU, but that could happen anyway when the futex is
	     acquired, so there's no need to check the nesting depth here.
	     See above for why relaxed MO is sufficient.  */
	  if (aconf.skip_lock_busy > 0)
	    atomic_store_relaxed (adapt_count, aconf.skip_lock_busy);
	}
      else if (status != _HTM_TBEGIN_TRANSIENT)
	{
	  /* A persistent abort (cc 1 or 3) indicates that a retry is
	     probably futile.  Use the normal locking now and for the
	     next couple of calls.
	     Be careful to avoid writing to the lock.  */
	  if (aconf.skip_trylock_internal_abort > 0)
	    *adapt_count = aconf.skip_trylock_internal_abort;
	}
      /* Could do some retries here.  */
    }

  /* Use normal locking as fallback path if the transaction does not
     succeed.  */
  return lll_trylock (*futex);
}
libc_hidden_def (__lll_trylock_elision)
