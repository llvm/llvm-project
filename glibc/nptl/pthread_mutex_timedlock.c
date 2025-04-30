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

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <sys/param.h>
#include <sys/time.h>
#include "pthreadP.h"
#include <atomic.h>
#include <lowlevellock.h>
#include <not-cancel.h>
#include <futex-internal.h>

#include <stap-probe.h>

int
__pthread_mutex_clocklock_common (pthread_mutex_t *mutex,
				  clockid_t clockid,
				  const struct __timespec64 *abstime)
{
  int oldval;
  pid_t id = THREAD_GETMEM (THREAD_SELF, tid);
  int result = 0;

  /* We must not check ABSTIME here.  If the thread does not block
     abstime must not be checked for a valid value.  */

  /* See concurrency notes regarding mutex type which is loaded from __kind
     in struct __pthread_mutex_s in sysdeps/nptl/bits/thread-shared-types.h.  */
  switch (__builtin_expect (PTHREAD_MUTEX_TYPE_ELISION (mutex),
			    PTHREAD_MUTEX_TIMED_NP))
    {
      /* Recursive mutex.  */
    case PTHREAD_MUTEX_RECURSIVE_NP|PTHREAD_MUTEX_ELISION_NP:
    case PTHREAD_MUTEX_RECURSIVE_NP:
      /* Check whether we already hold the mutex.  */
      if (mutex->__data.__owner == id)
	{
	  /* Just bump the counter.  */
	  if (__glibc_unlikely (mutex->__data.__count + 1 == 0))
	    /* Overflow of the counter.  */
	    return EAGAIN;

	  ++mutex->__data.__count;

	  goto out;
	}

      /* We have to get the mutex.  */
      result = __futex_clocklock64 (&mutex->__data.__lock, clockid, abstime,
                                    PTHREAD_MUTEX_PSHARED (mutex));

      if (result != 0)
	goto out;

      /* Only locked once so far.  */
      mutex->__data.__count = 1;
      break;

      /* Error checking mutex.  */
    case PTHREAD_MUTEX_ERRORCHECK_NP:
      /* Check whether we already hold the mutex.  */
      if (__glibc_unlikely (mutex->__data.__owner == id))
	return EDEADLK;

      /* Don't do lock elision on an error checking mutex.  */
      goto simple;

    case PTHREAD_MUTEX_TIMED_NP:
      FORCE_ELISION (mutex, goto elision);
    simple:
      /* Normal mutex.  */
      result = __futex_clocklock64 (&mutex->__data.__lock, clockid, abstime,
                                    PTHREAD_MUTEX_PSHARED (mutex));
      break;

    case PTHREAD_MUTEX_TIMED_ELISION_NP:
    elision: __attribute__((unused))
      /* Don't record ownership */
      return lll_clocklock_elision (mutex->__data.__lock,
				    mutex->__data.__spins,
				    clockid, abstime,
				    PTHREAD_MUTEX_PSHARED (mutex));


    case PTHREAD_MUTEX_ADAPTIVE_NP:
      if (lll_trylock (mutex->__data.__lock) != 0)
	{
	  int cnt = 0;
	  int max_cnt = MIN (max_adaptive_count (),
			     mutex->__data.__spins * 2 + 10);
	  do
	    {
	      if (cnt++ >= max_cnt)
		{
		  result = __futex_clocklock64 (&mutex->__data.__lock,
		                                clockid, abstime,
		                                PTHREAD_MUTEX_PSHARED (mutex));
		  break;
		}
	      atomic_spin_nop ();
	    }
	  while (lll_trylock (mutex->__data.__lock) != 0);

	  mutex->__data.__spins += (cnt - mutex->__data.__spins) / 8;
	}
      break;

    case PTHREAD_MUTEX_ROBUST_RECURSIVE_NP:
    case PTHREAD_MUTEX_ROBUST_ERRORCHECK_NP:
    case PTHREAD_MUTEX_ROBUST_NORMAL_NP:
    case PTHREAD_MUTEX_ROBUST_ADAPTIVE_NP:
      THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending,
		     &mutex->__data.__list.__next);
      /* We need to set op_pending before starting the operation.  Also
	 see comments at ENQUEUE_MUTEX.  */
      __asm ("" ::: "memory");

      oldval = mutex->__data.__lock;
      /* This is set to FUTEX_WAITERS iff we might have shared the
	 FUTEX_WAITERS flag with other threads, and therefore need to keep it
	 set to avoid lost wake-ups.  We have the same requirement in the
	 simple mutex algorithm.  */
      unsigned int assume_other_futex_waiters = 0;
      while (1)
	{
	  /* Try to acquire the lock through a CAS from 0 (not acquired) to
	     our TID | assume_other_futex_waiters.  */
	  if (__glibc_likely (oldval == 0))
	    {
	      oldval
	        = atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
	            id | assume_other_futex_waiters, 0);
	      if (__glibc_likely (oldval == 0))
		break;
	    }

	  if ((oldval & FUTEX_OWNER_DIED) != 0)
	    {
	      /* The previous owner died.  Try locking the mutex.  */
	      int newval = id | (oldval & FUTEX_WAITERS)
		  | assume_other_futex_waiters;

	      newval
		= atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
						       newval, oldval);
	      if (newval != oldval)
		{
		  oldval = newval;
		  continue;
		}

	      /* We got the mutex.  */
	      mutex->__data.__count = 1;
	      /* But it is inconsistent unless marked otherwise.  */
	      mutex->__data.__owner = PTHREAD_MUTEX_INCONSISTENT;

	      /* We must not enqueue the mutex before we have acquired it.
		 Also see comments at ENQUEUE_MUTEX.  */
	      __asm ("" ::: "memory");
	      ENQUEUE_MUTEX (mutex);
	      /* We need to clear op_pending after we enqueue the mutex.  */
	      __asm ("" ::: "memory");
	      THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);

	      /* Note that we deliberately exit here.  If we fall
		 through to the end of the function __nusers would be
		 incremented which is not correct because the old
		 owner has to be discounted.  */
	      return EOWNERDEAD;
	    }

	  /* Check whether we already hold the mutex.  */
	  if (__glibc_unlikely ((oldval & FUTEX_TID_MASK) == id))
	    {
	      int kind = PTHREAD_MUTEX_TYPE (mutex);
	      if (kind == PTHREAD_MUTEX_ROBUST_ERRORCHECK_NP)
		{
		  /* We do not need to ensure ordering wrt another memory
		     access.  Also see comments at ENQUEUE_MUTEX. */
		  THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending,
				 NULL);
		  return EDEADLK;
		}

	      if (kind == PTHREAD_MUTEX_ROBUST_RECURSIVE_NP)
		{
		  /* We do not need to ensure ordering wrt another memory
		     access.  */
		  THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending,
				 NULL);

		  /* Just bump the counter.  */
		  if (__glibc_unlikely (mutex->__data.__count + 1 == 0))
		    /* Overflow of the counter.  */
		    return EAGAIN;

		  ++mutex->__data.__count;

		  LIBC_PROBE (mutex_timedlock_acquired, 1, mutex);

		  return 0;
		}
	    }

	  /* We are about to block; check whether the timeout is invalid.  */
	  if (! valid_nanoseconds (abstime->tv_nsec))
	    return EINVAL;
	  /* Work around the fact that the kernel rejects negative timeout
	     values despite them being valid.  */
	  if (__glibc_unlikely (abstime->tv_sec < 0))
	    return ETIMEDOUT;

	  /* We cannot acquire the mutex nor has its owner died.  Thus, try
	     to block using futexes.  Set FUTEX_WAITERS if necessary so that
	     other threads are aware that there are potentially threads
	     blocked on the futex.  Restart if oldval changed in the
	     meantime.  */
	  if ((oldval & FUTEX_WAITERS) == 0)
	    {
	      if (atomic_compare_and_exchange_bool_acq (&mutex->__data.__lock,
							oldval | FUTEX_WAITERS,
							oldval)
		  != 0)
		{
		  oldval = mutex->__data.__lock;
		  continue;
		}
	      oldval |= FUTEX_WAITERS;
	    }

	  /* It is now possible that we share the FUTEX_WAITERS flag with
	     another thread; therefore, update assume_other_futex_waiters so
	     that we do not forget about this when handling other cases
	     above and thus do not cause lost wake-ups.  */
	  assume_other_futex_waiters |= FUTEX_WAITERS;

	  /* Block using the futex.  */
	  int err = __futex_abstimed_wait64 (
	      (unsigned int *) &mutex->__data.__lock,
	      oldval, clockid, abstime,
	      PTHREAD_ROBUST_MUTEX_PSHARED (mutex));
	  /* The futex call timed out.  */
	  if (err == ETIMEDOUT || err == EOVERFLOW)
	    return err;
	  /* Reload current lock value.  */
	  oldval = mutex->__data.__lock;
	}

      /* We have acquired the mutex; check if it is still consistent.  */
      if (__builtin_expect (mutex->__data.__owner
			    == PTHREAD_MUTEX_NOTRECOVERABLE, 0))
	{
	  /* This mutex is now not recoverable.  */
	  mutex->__data.__count = 0;
	  int private = PTHREAD_ROBUST_MUTEX_PSHARED (mutex);
	  lll_unlock (mutex->__data.__lock, private);
	  /* FIXME This violates the mutex destruction requirements.  See
	     __pthread_mutex_unlock_full.  */
	  THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
	  return ENOTRECOVERABLE;
	}

      mutex->__data.__count = 1;
      /* We must not enqueue the mutex before we have acquired it.
	 Also see comments at ENQUEUE_MUTEX.  */
      __asm ("" ::: "memory");
      ENQUEUE_MUTEX (mutex);
      /* We need to clear op_pending after we enqueue the mutex.  */
      __asm ("" ::: "memory");
      THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
      break;

    /* The PI support requires the Linux futex system call.  If that's not
       available, pthread_mutex_init should never have allowed the type to
       be set.  So it will get the default case for an invalid type.  */
#ifdef __NR_futex
    case PTHREAD_MUTEX_PI_RECURSIVE_NP:
    case PTHREAD_MUTEX_PI_ERRORCHECK_NP:
    case PTHREAD_MUTEX_PI_NORMAL_NP:
    case PTHREAD_MUTEX_PI_ADAPTIVE_NP:
    case PTHREAD_MUTEX_PI_ROBUST_RECURSIVE_NP:
    case PTHREAD_MUTEX_PI_ROBUST_ERRORCHECK_NP:
    case PTHREAD_MUTEX_PI_ROBUST_NORMAL_NP:
    case PTHREAD_MUTEX_PI_ROBUST_ADAPTIVE_NP:
      {
	/* Currently futex FUTEX_LOCK_PI operation only provides support for
	   CLOCK_REALTIME and trying to emulate by converting a
	   CLOCK_MONOTONIC to CLOCK_REALTIME will take in account possible
	   changes to the wall clock.  */
	if (__glibc_unlikely (clockid != CLOCK_REALTIME))
	  return EINVAL;

	int kind, robust;
	{
	  /* See concurrency notes regarding __kind in struct __pthread_mutex_s
	     in sysdeps/nptl/bits/thread-shared-types.h.  */
	  int mutex_kind = atomic_load_relaxed (&(mutex->__data.__kind));
	  kind = mutex_kind & PTHREAD_MUTEX_KIND_MASK_NP;
	  robust = mutex_kind & PTHREAD_MUTEX_ROBUST_NORMAL_NP;
	}

	if (robust)
	  {
	    /* Note: robust PI futexes are signaled by setting bit 0.  */
	    THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending,
			   (void *) (((uintptr_t) &mutex->__data.__list.__next)
				     | 1));
	    /* We need to set op_pending before starting the operation.  Also
	       see comments at ENQUEUE_MUTEX.  */
	    __asm ("" ::: "memory");
	  }

	oldval = mutex->__data.__lock;

	/* Check whether we already hold the mutex.  */
	if (__glibc_unlikely ((oldval & FUTEX_TID_MASK) == id))
	  {
	    if (kind == PTHREAD_MUTEX_ERRORCHECK_NP)
	      {
		/* We do not need to ensure ordering wrt another memory
		   access.  */
		THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
		return EDEADLK;
	      }

	    if (kind == PTHREAD_MUTEX_RECURSIVE_NP)
	      {
		/* We do not need to ensure ordering wrt another memory
		   access.  */
		THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);

		/* Just bump the counter.  */
		if (__glibc_unlikely (mutex->__data.__count + 1 == 0))
		  /* Overflow of the counter.  */
		  return EAGAIN;

		++mutex->__data.__count;

		LIBC_PROBE (mutex_timedlock_acquired, 1, mutex);

		return 0;
	      }
	  }

	oldval = atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
						      id, 0);

	if (oldval != 0)
	  {
	    /* The mutex is locked.  The kernel will now take care of
	       everything.  The timeout value must be a relative value.
	       Convert it.  */
	    int private = (robust
			   ? PTHREAD_ROBUST_MUTEX_PSHARED (mutex)
			   : PTHREAD_MUTEX_PSHARED (mutex));
	    int e = futex_lock_pi64 (&mutex->__data.__lock, abstime, private);
	    if (e == ETIMEDOUT)
	      return ETIMEDOUT;
	    else if (e == ESRCH || e == EDEADLK)
	      {
		assert (e != EDEADLK
			|| (kind != PTHREAD_MUTEX_ERRORCHECK_NP
			   && kind != PTHREAD_MUTEX_RECURSIVE_NP));
		/* ESRCH can happen only for non-robust PI mutexes where
		   the owner of the lock died.  */
		assert (e != ESRCH || !robust);

		/* Delay the thread until the timeout is reached. Then return
		   ETIMEDOUT.  */
		do
		  e = __futex_abstimed_wait64 (&(unsigned int){0}, 0, clockid,
					       abstime, private);
		while (e != ETIMEDOUT);
		return ETIMEDOUT;
	      }
	    else if (e != 0)
	      return e;

	    oldval = mutex->__data.__lock;

	    assert (robust || (oldval & FUTEX_OWNER_DIED) == 0);
	  }

	if (__glibc_unlikely (oldval & FUTEX_OWNER_DIED))
	  {
	    atomic_and (&mutex->__data.__lock, ~FUTEX_OWNER_DIED);

	    /* We got the mutex.  */
	    mutex->__data.__count = 1;
	    /* But it is inconsistent unless marked otherwise.  */
	    mutex->__data.__owner = PTHREAD_MUTEX_INCONSISTENT;

	    /* We must not enqueue the mutex before we have acquired it.
	       Also see comments at ENQUEUE_MUTEX.  */
	    __asm ("" ::: "memory");
	    ENQUEUE_MUTEX_PI (mutex);
	    /* We need to clear op_pending after we enqueue the mutex.  */
	    __asm ("" ::: "memory");
	    THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);

	    /* Note that we deliberately exit here.  If we fall
	       through to the end of the function __nusers would be
	       incremented which is not correct because the old owner
	       has to be discounted.  */
	    return EOWNERDEAD;
	  }

	if (robust
	    && __builtin_expect (mutex->__data.__owner
				 == PTHREAD_MUTEX_NOTRECOVERABLE, 0))
	  {
	    /* This mutex is now not recoverable.  */
	    mutex->__data.__count = 0;

	    futex_unlock_pi ((unsigned int *) &mutex->__data.__lock,
			     PTHREAD_ROBUST_MUTEX_PSHARED (mutex));

	    /* To the kernel, this will be visible after the kernel has
	       acquired the mutex in the syscall.  */
	    THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
	    return ENOTRECOVERABLE;
	  }

	mutex->__data.__count = 1;
	if (robust)
	  {
	    /* We must not enqueue the mutex before we have acquired it.
	       Also see comments at ENQUEUE_MUTEX.  */
	    __asm ("" ::: "memory");
	    ENQUEUE_MUTEX_PI (mutex);
	    /* We need to clear op_pending after we enqueue the mutex.  */
	    __asm ("" ::: "memory");
	    THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
	  }
	}
      break;
#endif  /* __NR_futex.  */

    case PTHREAD_MUTEX_PP_RECURSIVE_NP:
    case PTHREAD_MUTEX_PP_ERRORCHECK_NP:
    case PTHREAD_MUTEX_PP_NORMAL_NP:
    case PTHREAD_MUTEX_PP_ADAPTIVE_NP:
      {
	/* See concurrency notes regarding __kind in struct __pthread_mutex_s
	   in sysdeps/nptl/bits/thread-shared-types.h.  */
	int kind = atomic_load_relaxed (&(mutex->__data.__kind))
	  & PTHREAD_MUTEX_KIND_MASK_NP;

	oldval = mutex->__data.__lock;

	/* Check whether we already hold the mutex.  */
	if (mutex->__data.__owner == id)
	  {
	    if (kind == PTHREAD_MUTEX_ERRORCHECK_NP)
	      return EDEADLK;

	    if (kind == PTHREAD_MUTEX_RECURSIVE_NP)
	      {
		/* Just bump the counter.  */
		if (__glibc_unlikely (mutex->__data.__count + 1 == 0))
		  /* Overflow of the counter.  */
		  return EAGAIN;

		++mutex->__data.__count;

		LIBC_PROBE (mutex_timedlock_acquired, 1, mutex);

		return 0;
	      }
	  }

	int oldprio = -1, ceilval;
	do
	  {
	    int ceiling = (oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK)
			  >> PTHREAD_MUTEX_PRIO_CEILING_SHIFT;

	    if (__pthread_current_priority () > ceiling)
	      {
		result = EINVAL;
	      failpp:
		if (oldprio != -1)
		  __pthread_tpp_change_priority (oldprio, -1);
		return result;
	      }

	    result = __pthread_tpp_change_priority (oldprio, ceiling);
	    if (result)
	      return result;

	    ceilval = ceiling << PTHREAD_MUTEX_PRIO_CEILING_SHIFT;
	    oldprio = ceiling;

	    oldval
	      = atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
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
		  {
		    /* Reject invalid timeouts.  */
		    if (! valid_nanoseconds (abstime->tv_nsec))
		      {
			result = EINVAL;
			goto failpp;
		      }

		    int e = __futex_abstimed_wait64 (
		      (unsigned int *) &mutex->__data.__lock, ceilval | 2,
		      clockid, abstime, PTHREAD_MUTEX_PSHARED (mutex));
		    if (e == ETIMEDOUT || e == EOVERFLOW)
		      return e;
		  }
	      }
	    while (atomic_compare_and_exchange_val_acq (&mutex->__data.__lock,
							ceilval | 2, ceilval)
		   != ceilval);
	  }
	while ((oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK) != ceilval);

	assert (mutex->__data.__owner == 0);
	mutex->__data.__count = 1;
      }
      break;

    default:
      /* Correct code cannot set any other type.  */
      return EINVAL;
    }

  if (result == 0)
    {
      /* Record the ownership.  */
      mutex->__data.__owner = id;
      ++mutex->__data.__nusers;

      LIBC_PROBE (mutex_timedlock_acquired, 1, mutex);
    }

 out:
  return result;
}

int
___pthread_mutex_clocklock64 (pthread_mutex_t *mutex,
			      clockid_t clockid,
			      const struct __timespec64 *abstime)
{
  if (__glibc_unlikely (!futex_abstimed_supported_clockid (clockid)))
    return EINVAL;

  LIBC_PROBE (mutex_clocklock_entry, 3, mutex, clockid, abstime);
  return __pthread_mutex_clocklock_common (mutex, clockid, abstime);
}

#if __TIMESIZE == 64
strong_alias (___pthread_mutex_clocklock64, ___pthread_mutex_clocklock)
#else /* __TIMESPEC64 != 64 */
strong_alias (___pthread_mutex_clocklock64, __pthread_mutex_clocklock64)
libc_hidden_def (__pthread_mutex_clocklock64)

int
___pthread_mutex_clocklock (pthread_mutex_t *mutex,
			    clockid_t clockid,
			    const struct timespec *abstime)
{
  struct __timespec64 ts64 = valid_timespec_to_timespec64 (*abstime);

  return ___pthread_mutex_clocklock64 (mutex, clockid, &ts64);
}
#endif /* __TIMESPEC64 != 64 */
libc_hidden_ver (___pthread_mutex_clocklock, __pthread_mutex_clocklock)
#ifndef SHARED
strong_alias (___pthread_mutex_clocklock, __pthread_mutex_clocklock)
#endif
versioned_symbol (libc, ___pthread_mutex_clocklock,
		  pthread_mutex_clocklock, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_30, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutex_clocklock,
	       pthread_mutex_clocklock, GLIBC_2_30);
#endif

int
___pthread_mutex_timedlock64 (pthread_mutex_t *mutex,
			     const struct __timespec64 *abstime)
{
  LIBC_PROBE (mutex_timedlock_entry, 2, mutex, abstime);
  return __pthread_mutex_clocklock_common (mutex, CLOCK_REALTIME, abstime);
}

#if __TIMESIZE == 64
strong_alias (___pthread_mutex_timedlock64, ___pthread_mutex_timedlock)
#else /* __TIMESPEC64 != 64 */
strong_alias (___pthread_mutex_timedlock64, __pthread_mutex_timedlock64);
libc_hidden_def (__pthread_mutex_timedlock64)

int
___pthread_mutex_timedlock (pthread_mutex_t *mutex,
			   const struct timespec *abstime)
{
  struct __timespec64 ts64 = valid_timespec_to_timespec64 (*abstime);

  return __pthread_mutex_timedlock64 (mutex, &ts64);
}
#endif /* __TIMESPEC64 != 64 */
versioned_symbol (libc, ___pthread_mutex_timedlock,
		  pthread_mutex_timedlock, GLIBC_2_34);
libc_hidden_ver (___pthread_mutex_timedlock, __pthread_mutex_timedlock)
#ifndef SHARED
strong_alias (___pthread_mutex_timedlock, __pthread_mutex_timedlock)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutex_timedlock,
	       pthread_mutex_timedlock, GLIBC_2_2);
#endif
