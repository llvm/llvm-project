/* elision-unlock.c: Commit an elided pthread lock.
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

#include "pthreadP.h"
#include "lowlevellock.h"
#include "htm.h"

int
__lll_unlock_elision (int *lock, short *adapt_count, int pshared)
{
  /* When the lock was free we're in a transaction.  */
  if (*lock == 0)
    __libc_tend (0);
  else
    {
      /* Update adapt_count in the critical section to prevent a
	 write-after-destroy error as mentioned in BZ 20822.  The
	 following update of adapt_count has to be contained within
	 the critical region of the fall-back lock in order to not violate
	 the mutex destruction requirements.  */
      short __tmp = atomic_load_relaxed (adapt_count);
      if (__tmp > 0)
	atomic_store_relaxed (adapt_count, __tmp - 1);

      lll_unlock ((*lock), pshared);
    }
  return 0;
}
libc_hidden_def (__lll_unlock_elision)
