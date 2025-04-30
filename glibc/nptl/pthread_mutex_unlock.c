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
#include <stdlib.h>
#include "pthreadP.h"
#include <lowlevellock.h>
#include <stap-probe.h>
#include <futex-internal.h>
#include <shlib-compat.h>

static int
__pthread_mutex_unlock_full (pthread_mutex_t *mutex, int decr)
     __attribute_noinline__;

/* lll_lock with single-thread optimization.  */
static inline void
lll_mutex_unlock_optimized (pthread_mutex_t *mutex)
{
  /* The single-threaded optimization is only valid for private
     mutexes.  For process-shared mutexes, the mutex could be in a
     shared mapping, so synchronization with another process is needed
     even without any threads.  */
  int private = PTHREAD_MUTEX_PSHARED (mutex);
  if (private == LLL_PRIVATE && SINGLE_THREAD_P)
    mutex->__data.__lock = 0;
  else
    lll_unlock (mutex->__data.__lock, private);
}

int
__pthread_mutex_unlock_usercnt (pthread_mutex_t *mutex, int decr)
{
  /* See concurrency notes regarding mutex type which is loaded from __kind
     in struct __pthread_mutex_s in sysdeps/nptl/bits/thread-shared-types.h.  */
  int type = PTHREAD_MUTEX_TYPE_ELISION (mutex);
  if (__builtin_expect (type
			& ~(PTHREAD_MUTEX_KIND_MASK_NP
			    |PTHREAD_MUTEX_ELISION_FLAGS_NP), 0))
    return __pthread_mutex_unlock_full (mutex, decr);

  if (__builtin_expect (type, PTHREAD_MUTEX_TIMED_NP)
      == PTHREAD_MUTEX_TIMED_NP)
    {
      /* Always reset the owner field.  */
    normal:
      mutex->__data.__owner = 0;
      if (decr)
	/* One less user.  */
	--mutex->__data.__nusers;

      /* Unlock.  */
      lll_mutex_unlock_optimized (mutex);

      LIBC_PROBE (mutex_release, 1, mutex);

      return 0;
    }
  else if (__glibc_likely (type == PTHREAD_MUTEX_TIMED_ELISION_NP))
    {
      /* Don't reset the owner/users fields for elision.  */
      return lll_unlock_elision (mutex->__data.__lock, mutex->__data.__elision,
				      PTHREAD_MUTEX_PSHARED (mutex));
    }
  else if (__builtin_expect (PTHREAD_MUTEX_TYPE (mutex)
			      == PTHREAD_MUTEX_RECURSIVE_NP, 1))
    {
      /* Recursive mutex.  */
      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid))
	return EPERM;

      if (--mutex->__data.__count != 0)
	/* We still hold the mutex.  */
	return 0;
      goto normal;
    }
  else if (__builtin_expect (PTHREAD_MUTEX_TYPE (mutex)
			      == PTHREAD_MUTEX_ADAPTIVE_NP, 1))
    goto normal;
  else
    {
      /* Error checking mutex.  */
      assert (type == PTHREAD_MUTEX_ERRORCHECK_NP);
      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid)
	  || ! lll_islocked (mutex->__data.__lock))
	return EPERM;
      goto normal;
    }
}
libc_hidden_def (__pthread_mutex_unlock_usercnt)


static int
__pthread_mutex_unlock_full (pthread_mutex_t *mutex, int decr)
{
  int newowner = 0;
  int private;

  switch (PTHREAD_MUTEX_TYPE (mutex))
    {
    case PTHREAD_MUTEX_ROBUST_RECURSIVE_NP:
      /* Recursive mutex.  */
      if ((mutex->__data.__lock & FUTEX_TID_MASK)
	  == THREAD_GETMEM (THREAD_SELF, tid)
	  && __builtin_expect (mutex->__data.__owner
			       == PTHREAD_MUTEX_INCONSISTENT, 0))
	{
	  if (--mutex->__data.__count != 0)
	    /* We still hold the mutex.  */
	    return ENOTRECOVERABLE;

	  goto notrecoverable;
	}

      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid))
	return EPERM;

      if (--mutex->__data.__count != 0)
	/* We still hold the mutex.  */
	return 0;

      goto robust;

    case PTHREAD_MUTEX_ROBUST_ERRORCHECK_NP:
    case PTHREAD_MUTEX_ROBUST_NORMAL_NP:
    case PTHREAD_MUTEX_ROBUST_ADAPTIVE_NP:
      if ((mutex->__data.__lock & FUTEX_TID_MASK)
	  != THREAD_GETMEM (THREAD_SELF, tid)
	  || ! lll_islocked (mutex->__data.__lock))
	return EPERM;

      /* If the previous owner died and the caller did not succeed in
	 making the state consistent, mark the mutex as unrecoverable
	 and make all waiters.  */
      if (__builtin_expect (mutex->__data.__owner
			    == PTHREAD_MUTEX_INCONSISTENT, 0))
      notrecoverable:
	newowner = PTHREAD_MUTEX_NOTRECOVERABLE;

    robust:
      /* Remove mutex from the list.  */
      THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending,
		     &mutex->__data.__list.__next);
      /* We must set op_pending before we dequeue the mutex.  Also see
	 comments at ENQUEUE_MUTEX.  */
      __asm ("" ::: "memory");
      DEQUEUE_MUTEX (mutex);

      mutex->__data.__owner = newowner;
      if (decr)
	/* One less user.  */
	--mutex->__data.__nusers;

      /* Unlock by setting the lock to 0 (not acquired); if the lock had
	 FUTEX_WAITERS set previously, then wake any waiters.
         The unlock operation must be the last access to the mutex to not
         violate the mutex destruction requirements (see __lll_unlock).  */
      private = PTHREAD_ROBUST_MUTEX_PSHARED (mutex);
      if (__glibc_unlikely ((atomic_exchange_rel (&mutex->__data.__lock, 0)
			     & FUTEX_WAITERS) != 0))
	futex_wake ((unsigned int *) &mutex->__data.__lock, 1, private);

      /* We must clear op_pending after we release the mutex.
	 FIXME However, this violates the mutex destruction requirements
	 because another thread could acquire the mutex, destroy it, and
	 reuse the memory for something else; then, if this thread crashes,
	 and the memory happens to have a value equal to the TID, the kernel
	 will believe it is still related to the mutex (which has been
	 destroyed already) and will modify some other random object.  */
      __asm ("" ::: "memory");
      THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
      break;

    /* The PI support requires the Linux futex system call.  If that's not
       available, pthread_mutex_init should never have allowed the type to
       be set.  So it will get the default case for an invalid type.  */
#ifdef __NR_futex
    case PTHREAD_MUTEX_PI_RECURSIVE_NP:
      /* Recursive mutex.  */
      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid))
	return EPERM;

      if (--mutex->__data.__count != 0)
	/* We still hold the mutex.  */
	return 0;
      goto continue_pi_non_robust;

    case PTHREAD_MUTEX_PI_ROBUST_RECURSIVE_NP:
      /* Recursive mutex.  */
      if ((mutex->__data.__lock & FUTEX_TID_MASK)
	  == THREAD_GETMEM (THREAD_SELF, tid)
	  && __builtin_expect (mutex->__data.__owner
			       == PTHREAD_MUTEX_INCONSISTENT, 0))
	{
	  if (--mutex->__data.__count != 0)
	    /* We still hold the mutex.  */
	    return ENOTRECOVERABLE;

	  goto pi_notrecoverable;
	}

      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid))
	return EPERM;

      if (--mutex->__data.__count != 0)
	/* We still hold the mutex.  */
	return 0;

      goto continue_pi_robust;

    case PTHREAD_MUTEX_PI_ERRORCHECK_NP:
    case PTHREAD_MUTEX_PI_NORMAL_NP:
    case PTHREAD_MUTEX_PI_ADAPTIVE_NP:
    case PTHREAD_MUTEX_PI_ROBUST_ERRORCHECK_NP:
    case PTHREAD_MUTEX_PI_ROBUST_NORMAL_NP:
    case PTHREAD_MUTEX_PI_ROBUST_ADAPTIVE_NP:
      if ((mutex->__data.__lock & FUTEX_TID_MASK)
	  != THREAD_GETMEM (THREAD_SELF, tid)
	  || ! lll_islocked (mutex->__data.__lock))
	return EPERM;

      /* If the previous owner died and the caller did not succeed in
	 making the state consistent, mark the mutex as unrecoverable
	 and make all waiters.  */
      /* See concurrency notes regarding __kind in struct __pthread_mutex_s
	 in sysdeps/nptl/bits/thread-shared-types.h.  */
      if ((atomic_load_relaxed (&(mutex->__data.__kind))
	   & PTHREAD_MUTEX_ROBUST_NORMAL_NP) != 0
	  && __builtin_expect (mutex->__data.__owner
			       == PTHREAD_MUTEX_INCONSISTENT, 0))
      pi_notrecoverable:
       newowner = PTHREAD_MUTEX_NOTRECOVERABLE;

      /* See concurrency notes regarding __kind in struct __pthread_mutex_s
	 in sysdeps/nptl/bits/thread-shared-types.h.  */
      if ((atomic_load_relaxed (&(mutex->__data.__kind))
	   & PTHREAD_MUTEX_ROBUST_NORMAL_NP) != 0)
	{
	continue_pi_robust:
	  /* Remove mutex from the list.
	     Note: robust PI futexes are signaled by setting bit 0.  */
	  THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending,
			 (void *) (((uintptr_t) &mutex->__data.__list.__next)
				   | 1));
	  /* We must set op_pending before we dequeue the mutex.  Also see
	     comments at ENQUEUE_MUTEX.  */
	  __asm ("" ::: "memory");
	  DEQUEUE_MUTEX (mutex);
	}

    continue_pi_non_robust:
      mutex->__data.__owner = newowner;
      if (decr)
	/* One less user.  */
	--mutex->__data.__nusers;

      /* Unlock.  Load all necessary mutex data before releasing the mutex
	 to not violate the mutex destruction requirements (see
	 lll_unlock).  */
      /* See concurrency notes regarding __kind in struct __pthread_mutex_s
	 in sysdeps/nptl/bits/thread-shared-types.h.  */
      int robust = atomic_load_relaxed (&(mutex->__data.__kind))
	& PTHREAD_MUTEX_ROBUST_NORMAL_NP;
      private = (robust
		 ? PTHREAD_ROBUST_MUTEX_PSHARED (mutex)
		 : PTHREAD_MUTEX_PSHARED (mutex));
      /* Unlock the mutex using a CAS unless there are futex waiters or our
	 TID is not the value of __lock anymore, in which case we let the
	 kernel take care of the situation.  Use release MO in the CAS to
	 synchronize with acquire MO in lock acquisitions.  */
      int l = atomic_load_relaxed (&mutex->__data.__lock);
      do
	{
	  if (((l & FUTEX_WAITERS) != 0)
	      || (l != THREAD_GETMEM (THREAD_SELF, tid)))
	    {
	      futex_unlock_pi ((unsigned int *) &mutex->__data.__lock,
			       private);
	      break;
	    }
	}
      while (!atomic_compare_exchange_weak_release (&mutex->__data.__lock,
						    &l, 0));

      /* This happens after the kernel releases the mutex but violates the
	 mutex destruction requirements; see comments in the code handling
	 PTHREAD_MUTEX_ROBUST_NORMAL_NP.  */
      THREAD_SETMEM (THREAD_SELF, robust_head.list_op_pending, NULL);
      break;
#endif  /* __NR_futex.  */

    case PTHREAD_MUTEX_PP_RECURSIVE_NP:
      /* Recursive mutex.  */
      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid))
	return EPERM;

      if (--mutex->__data.__count != 0)
	/* We still hold the mutex.  */
	return 0;
      goto pp;

    case PTHREAD_MUTEX_PP_ERRORCHECK_NP:
      /* Error checking mutex.  */
      if (mutex->__data.__owner != THREAD_GETMEM (THREAD_SELF, tid)
	  || (mutex->__data.__lock & ~ PTHREAD_MUTEX_PRIO_CEILING_MASK) == 0)
	return EPERM;
      /* FALLTHROUGH */

    case PTHREAD_MUTEX_PP_NORMAL_NP:
    case PTHREAD_MUTEX_PP_ADAPTIVE_NP:
      /* Always reset the owner field.  */
    pp:
      mutex->__data.__owner = 0;

      if (decr)
	/* One less user.  */
	--mutex->__data.__nusers;

      /* Unlock.  Use release MO in the CAS to synchronize with acquire MO in
	 lock acquisitions.  */
      int newval;
      int oldval = atomic_load_relaxed (&mutex->__data.__lock);
      do
	{
	  newval = oldval & PTHREAD_MUTEX_PRIO_CEILING_MASK;
	}
      while (!atomic_compare_exchange_weak_release (&mutex->__data.__lock,
						    &oldval, newval));

      if ((oldval & ~PTHREAD_MUTEX_PRIO_CEILING_MASK) > 1)
	futex_wake ((unsigned int *)&mutex->__data.__lock, 1,
		    PTHREAD_MUTEX_PSHARED (mutex));

      int oldprio = newval >> PTHREAD_MUTEX_PRIO_CEILING_SHIFT;

      LIBC_PROBE (mutex_release, 1, mutex);

      return __pthread_tpp_change_priority (oldprio, -1);

    default:
      /* Correct code cannot set any other type.  */
      return EINVAL;
    }

  LIBC_PROBE (mutex_release, 1, mutex);
  return 0;
}


int
___pthread_mutex_unlock (pthread_mutex_t *mutex)
{
  return __pthread_mutex_unlock_usercnt (mutex, 1);
}
libc_hidden_ver (___pthread_mutex_unlock, __pthread_mutex_unlock)
#ifndef SHARED
strong_alias (___pthread_mutex_unlock, __pthread_mutex_unlock)
#endif
versioned_symbol (libpthread, ___pthread_mutex_unlock, pthread_mutex_unlock,
		  GLIBC_2_0);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutex_unlock, __pthread_mutex_unlock,
	       GLIBC_2_0);
#endif
