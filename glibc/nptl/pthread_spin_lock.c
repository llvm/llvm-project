/* pthread_spin_lock -- lock a spin lock.  Generic version.
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

#include <atomic.h>
#include "pthreadP.h"
#include <shlib-compat.h>

int
__pthread_spin_lock (pthread_spinlock_t *lock)
{
  int val = 0;

  /* We assume that the first try mostly will be successful, thus we use
     atomic_exchange if it is not implemented by a CAS loop (we also assume
     that atomic_exchange can be faster if it succeeds, see
     ATOMIC_EXCHANGE_USES_CAS).  Otherwise, we use a weak CAS and not an
     exchange so we bail out after the first failed attempt to change the
     state.  For the subsequent attempts we use atomic_compare_and_exchange
     after we observe that the lock is not acquired.
     See also comment in pthread_spin_trylock.
     We use acquire MO to synchronize-with the release MO store in
     pthread_spin_unlock, and thus ensure that prior critical sections
     happen-before this critical section.  */
#if ! ATOMIC_EXCHANGE_USES_CAS
  /* Try to acquire the lock with an exchange instruction as this architecture
     has such an instruction and we assume it is faster than a CAS.
     The acquisition succeeds if the lock is not in an acquired state.  */
  if (__glibc_likely (atomic_exchange_acquire (lock, 1) == 0))
    return 0;
#else
  /* Try to acquire the lock with a CAS instruction as this architecture
     has no exchange instruction.  The acquisition succeeds if the lock is not
     acquired.  */
  if (__glibc_likely (atomic_compare_exchange_weak_acquire (lock, &val, 1)))
    return 0;
#endif

  do
    {
      /* The lock is contended and we need to wait.  Going straight back
	 to cmpxchg is not a good idea on many targets as that will force
	 expensive memory synchronizations among processors and penalize other
	 running threads.
	 There is no technical reason for throwing in a CAS every now and then,
	 and so far we have no evidence that it can improve performance.
	 If that would be the case, we have to adjust other spin-waiting loops
	 elsewhere, too!
	 Thus we use relaxed MO reads until we observe the lock to not be
	 acquired anymore.  */
      do
	{
	  /* TODO Back-off.  */

	  atomic_spin_nop ();

	  val = atomic_load_relaxed (lock);
	}
      while (val != 0);

      /* We need acquire memory order here for the same reason as mentioned
	 for the first try to lock the spinlock.  */
    }
  while (!atomic_compare_exchange_weak_acquire (lock, &val, 1));

  return 0;
}
versioned_symbol (libc, __pthread_spin_lock, pthread_spin_lock, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_lock, pthread_spin_lock, GLIBC_2_2);
#endif
