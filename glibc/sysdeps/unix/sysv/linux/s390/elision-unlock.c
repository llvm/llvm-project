/* Commit an elided pthread lock.
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

#include <pthreadP.h>
#include <lowlevellock.h>
#include <htm.h>

int
__lll_unlock_elision(int *futex, short *adapt_count, int private)
{
  /* If the lock is free, we elided the lock earlier.  This does not
     necessarily mean that we are in a transaction, because the user code may
     have closed the transaction, but that is impossible to detect reliably.
     Relaxed MO access to futex is sufficient because a correct program
     will only release a lock it has acquired; therefore, it must either
     changed the futex word's value to something !=0 or it must have used
     elision; these are actions by the same thread, so these actions are
     sequenced-before the relaxed load (and thus also happens-before the
     relaxed load).  Therefore, relaxed MO is sufficient.  */
  if (atomic_load_relaxed (futex) == 0)
    {
      __libc_tend ();
    }
  else
    {
      /* Update the adapt_count while unlocking before completing the critical
	 section.  adapt_count is accessed concurrently outside of a
	 transaction or a critical section (e.g. in elision-lock.c). So we need
	 to use atomic accesses.  However, the value of adapt_count is just a
	 hint, so relaxed MO accesses are sufficient.
	 If adapt_count would be decremented while locking, multiple
	 CPUs, trying to lock the acquired mutex, will decrement adapt_count to
	 zero and another CPU will try to start a transaction, which will be
	 immediately aborted as the mutex is locked.
	 The update of adapt_count is done before releasing the lock as POSIX'
	 mutex destruction requirements disallow accesses to the mutex after it
	 has been released and thus could have been acquired or destroyed by
	 another thread.  */
      short adapt_count_val = atomic_load_relaxed (adapt_count);
      if (adapt_count_val > 0)
	atomic_store_relaxed (adapt_count, adapt_count_val - 1);

      lll_unlock ((*futex), private);
    }
  return 0;
}
libc_hidden_def (__lll_unlock_elision)
