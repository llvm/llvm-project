/* Set current priority ceiling of pthread_mutex_t.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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

#include <stdbool.h>
#include <errno.h>
#include <pthreadP.h>
#include <atomic.h>
#include <futex-internal.h>
#include <shlib-compat.h>

int
__pthread_mutex_setprioceiling (pthread_mutex_t *mutex, int prioceiling,
				int *old_ceiling)
{
  /* See concurrency notes regarding __kind in struct __pthread_mutex_s
     in sysdeps/nptl/bits/thread-shared-types.h.  */
  if ((atomic_load_relaxed (&(mutex->__data.__kind))
       & PTHREAD_MUTEX_PRIO_PROTECT_NP) == 0)
    return EINVAL;

  /* See __init_sched_fifo_prio.  */
  if (atomic_load_relaxed (&__sched_fifo_min_prio) == -1
      || atomic_load_relaxed (&__sched_fifo_max_prio) == -1)
    __init_sched_fifo_prio ();

  if (__glibc_unlikely (prioceiling
			< atomic_load_relaxed (&__sched_fifo_min_prio))
      || __glibc_unlikely (prioceiling
			   > atomic_load_relaxed (&__sched_fifo_max_prio))
      || __glibc_unlikely ((prioceiling
			    & (PTHREAD_MUTEXATTR_PRIO_CEILING_MASK
			       >> PTHREAD_MUTEXATTR_PRIO_CEILING_SHIFT))
			   != prioceiling))
    return EINVAL;

  /* Check whether we already hold the mutex.  */
  bool locked = false;
  int kind = PTHREAD_MUTEX_TYPE (mutex);
  if (mutex->__data.__owner == THREAD_GETMEM (THREAD_SELF, tid))
    {
      if (kind == PTHREAD_MUTEX_PP_ERRORCHECK_NP)
	return EDEADLK;

      if (kind == PTHREAD_MUTEX_PP_RECURSIVE_NP)
	locked = true;
    }

  int oldval = mutex->__data.__lock;
  if (! locked)
    do
      {
	/* Need to lock the mutex, but without obeying the priority
	   protect protocol.  */
	int ceilval = (oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK);

	oldval = atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
						      ceilval | 1, ceilval);
	if (oldval == ceilval)
	  break;

	do
	  {
	    oldval
	      = atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
						     ceilval | 2,
						     ceilval | 1);

	    if ((oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK) != ceilval)
	      break;

	    if (oldval != ceilval)
	      futex_wait ((unsigned int *) &mutex->__data.__lock, ceilval | 2,
			  PTHREAD_MUTEX_PSHARED (mutex));
	  }
	while (atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
						    ceilval | 2, ceilval)
	       != ceilval);

	if ((oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK) != ceilval)
	  continue;
      }
    while (0);

  int oldprio = (oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK)
		>> PTHREAD_MUTEX_PRIO_CEILING_SHIFT;
  if (locked)
    {
      int ret = __pthread_tpp_change_priority (oldprio, prioceiling);
      if (ret)
	return ret;
    }

  if (old_ceiling != NULL)
    *old_ceiling = oldprio;

  int newlock = 0;
  if (locked)
    newlock = (mutex->__data.__lock & ~PTHREAD_MUTEX_PRIO_CEILING_MASK);
  mutex->__data.__lock = newlock
			 | (prioceiling << PTHREAD_MUTEX_PRIO_CEILING_SHIFT);
  atomic_full_barrier ();

  futex_wake ((unsigned int *)&mutex->__data.__lock, INT_MAX,
	      PTHREAD_MUTEX_PSHARED (mutex));

  return 0;
}
versioned_symbol (libc, __pthread_mutex_setprioceiling,
		  pthread_mutex_setprioceiling, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_4, GLIBC_2_34)
compat_symbol (libpthread, __pthread_mutex_setprioceiling,
               pthread_mutex_setprioceiling, GLIBC_2_4);
#endif
