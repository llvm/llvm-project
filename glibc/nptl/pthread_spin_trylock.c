/* pthread_spin_trylock -- trylock a spin lock.  Generic version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <atomic.h>
#include "pthreadP.h"
#include <shlib-compat.h>

int
__pthread_spin_trylock (pthread_spinlock_t *lock)
{
  /* For the spin try lock, we have the following possibilities:

     1) If we assume that trylock will most likely succeed in practice:
     * We just do an exchange.

     2) If we want to bias towards cases where trylock succeeds, but don't
     rule out contention:
     * If exchange is not implemented by a CAS loop, and exchange is faster
     than CAS, do an exchange.
     * If exchange is implemented by a CAS loop, use a weak CAS and not an
     exchange so we bail out after the first failed attempt to change the state.

     3) If we expect contention to be likely:
     * If CAS always brings the cache line into an exclusive state even if the
     spinlock is already acquired, then load the value first with
     atomic_load_relaxed and test if lock is not acquired. Then do 2).

     We assume that 2) is the common case, and that this won't be slower than
     1) in the common case.

     We use acquire MO to synchronize-with the release MO store in
     pthread_spin_unlock, and thus ensure that prior critical sections
     happen-before this critical section.  */
#if ! ATOMIC_EXCHANGE_USES_CAS
  /* Try to acquire the lock with an exchange instruction as this architecture
     has such an instruction and we assume it is faster than a CAS.
     The acquisition succeeds if the lock is not in an acquired state.  */
  if (atomic_exchange_acquire (lock, 1) == 0)
    return 0;
#else
  /* Try to acquire the lock with a CAS instruction as this architecture
     has no exchange instruction.  The acquisition succeeds if the lock is not
     acquired.  */
  do
    {
      int val = 0;
      if (atomic_compare_exchange_weak_acquire (lock, &val, 1))
	return 0;
    }
  /* atomic_compare_exchange_weak_acquire can fail spuriously.  Whereas
     C++11 and C11 make it clear that trylock operations can fail spuriously,
     POSIX does not explicitly specify this; it only specifies that failing
     synchronization operations do not need to have synchronization effects
     themselves, but a spurious failure is something that could contradict a
     happens-before established earlier (e.g., that we need to observe that
     the lock is acquired).  Therefore, we emulate a strong CAS by simply
     checking with a relaxed MO load that the lock is really acquired before
     returning EBUSY; the additional overhead this may cause is on the slow
     path.  */
  while (atomic_load_relaxed (lock) == 0);
#endif

  return EBUSY;
}
versioned_symbol (libc, __pthread_spin_trylock, pthread_spin_trylock,
		  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_trylock, pthread_spin_trylock,
	       GLIBC_2_2);
#endif
