/* Low-level lock implementation.  Generic futex-based version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _LOWLEVELLOCK_H
#define _LOWLEVELLOCK_H	1

#include <atomic.h>
#include <elision-conf.h>
#include <lowlevellock-futex.h>
#include <time.h>

/* Low-level locks use a combination of atomic operations (to acquire and
   release lock ownership) and futex operations (to block until the state
   of a lock changes).  A lock can be in one of three states:
   0:  not acquired,
   1:  acquired with no waiters; no other threads are blocked or about to block
       for changes to the lock state,
   >1: acquired, possibly with waiters; there may be other threads blocked or
       about to block for changes to the lock state.

   We expect that the common case is an uncontended lock, so we just need
   to transition the lock between states 0 and 1; releasing the lock does
   not need to wake any other blocked threads.  If the lock is contended
   and a thread decides to block using a futex operation, then this thread
   needs to first change the state to >1; if this state is observed during
   lock release, the releasing thread will wake one of the potentially
   blocked threads.

   Much of this code takes a 'private' parameter.  This may be:
   LLL_PRIVATE: lock only shared within a process
   LLL_SHARED:  lock may be shared across processes.

   Condition variables contain an optimization for broadcasts that requeues
   waiting threads on a lock's futex.  Therefore, there is a special
   variant of the locks (whose name contains "cond") that makes sure to
   always set the lock state to >1 and not just 1.

   Robust locks set the lock to the id of the owner.  This allows detection
   of the case where the owner exits without releasing the lock.  Flags are
   OR'd with the owner id to record additional information about lock state.
   Therefore the states of robust locks are:
    0: not acquired
   id: acquired (by user identified by id & FUTEX_TID_MASK)

   The following flags may be set in the robust lock value:
   FUTEX_WAITERS     - possibly has waiters
   FUTEX_OWNER_DIED  - owning user has exited without releasing the futex.  */


/* If LOCK is 0 (not acquired), set to 1 (acquired with no waiters) and return
   0.  Otherwise leave lock unchanged and return non-zero to indicate that the
   lock was not acquired.  */
#define __lll_trylock(lock)	\
  __glibc_unlikely (atomic_compare_and_exchange_bool_acq ((lock), 1, 0))
#define lll_trylock(lock)	\
   __lll_trylock (&(lock))

/* If LOCK is 0 (not acquired), set to 2 (acquired, possibly with waiters) and
   return 0.  Otherwise leave lock unchanged and return non-zero to indicate
   that the lock was not acquired.  */
#define lll_cond_trylock(lock)	\
  __glibc_unlikely (atomic_compare_and_exchange_bool_acq (&(lock), 2, 0))

extern void __lll_lock_wait_private (int *futex);
libc_hidden_proto (__lll_lock_wait_private)
extern void __lll_lock_wait (int *futex, int private);
libc_hidden_proto (__lll_lock_wait)

/* This is an expression rather than a statement even though its value is
   void, so that it can be used in a comma expression or as an expression
   that's cast to void.  */
/* The inner conditional compiles to a call to __lll_lock_wait_private if
   private is known at compile time to be LLL_PRIVATE, and to a call to
   __lll_lock_wait otherwise.  */
/* If FUTEX is 0 (not acquired), set to 1 (acquired with no waiters) and
   return.  Otherwise, ensure that it is >1 (acquired, possibly with waiters)
   and then block until we acquire the lock, at which point FUTEX will still be
   >1.  The lock is always acquired on return.  */
#define __lll_lock(futex, private)                                      \
  ((void)                                                               \
   ({                                                                   \
     int *__futex = (futex);                                            \
     if (__glibc_unlikely                                               \
         (atomic_compare_and_exchange_bool_acq (__futex, 1, 0)))        \
       {                                                                \
         if (__builtin_constant_p (private) && (private) == LLL_PRIVATE) \
           __lll_lock_wait_private (__futex);                           \
         else                                                           \
           __lll_lock_wait (__futex, private);                          \
       }                                                                \
   }))
#define lll_lock(futex, private)	\
  __lll_lock (&(futex), private)


/* This is an expression rather than a statement even though its value is
   void, so that it can be used in a comma expression or as an expression
   that's cast to void.  */
/* Unconditionally set FUTEX to 2 (acquired, possibly with waiters).  If FUTEX
   was 0 (not acquired) then return.  Otherwise, block until the lock is
   acquired, at which point FUTEX is 2 (acquired, possibly with waiters).  The
   lock is always acquired on return.  */
#define __lll_cond_lock(futex, private)                                 \
  ((void)                                                               \
   ({                                                                   \
     int *__futex = (futex);                                            \
     if (__glibc_unlikely (atomic_exchange_acq (__futex, 2) != 0))      \
       __lll_lock_wait (__futex, private);                              \
   }))
#define lll_cond_lock(futex, private) __lll_cond_lock (&(futex), private)


extern void __lll_lock_wake_private (int *futex);
libc_hidden_proto (__lll_lock_wake_private)
extern void __lll_lock_wake (int *futex, int private);
libc_hidden_proto (__lll_lock_wake)

/* This is an expression rather than a statement even though its value is
   void, so that it can be used in a comma expression or as an expression
   that's cast to void.  */
/* Unconditionally set FUTEX to 0 (not acquired), releasing the lock.  If FUTEX
   was >1 (acquired, possibly with waiters), then wake any waiters.  The waiter
   that acquires the lock will set FUTEX to >1.
   Evaluate PRIVATE before releasing the lock so that we do not violate the
   mutex destruction requirements.  Specifically, we need to ensure that
   another thread can destroy the mutex (and reuse its memory) once it
   acquires the lock and when there will be no further lock acquisitions;
   thus, we must not access the lock after releasing it, or those accesses
   could be concurrent with mutex destruction or reuse of the memory.  */
#define __lll_unlock(futex, private)					\
  ((void)								\
  ({									\
     int *__futex = (futex);						\
     int __private = (private);						\
     int __oldval = atomic_exchange_rel (__futex, 0);			\
     if (__glibc_unlikely (__oldval > 1))				\
       {								\
         if (__builtin_constant_p (private) && (private) == LLL_PRIVATE) \
           __lll_lock_wake_private (__futex);                           \
         else                                                           \
           __lll_lock_wake (__futex, __private);			\
       }								\
   }))
#define lll_unlock(futex, private)	\
  __lll_unlock (&(futex), private)


#define lll_islocked(futex) \
  ((futex) != LLL_LOCK_INITIALIZER)


/* Our internal lock implementation is identical to the binary-compatible
   mutex implementation. */

/* Initializers for lock.  */
#define LLL_LOCK_INITIALIZER		(0)
#define LLL_LOCK_INITIALIZER_LOCKED	(1)

/* Elision support.  */

#if ENABLE_ELISION_SUPPORT
/* Force elision for all new locks.  This is used to decide whether
   existing DEFAULT locks should be automatically upgraded to elision
   in pthread_mutex_lock.  Disabled for suid programs.  Only used when
   elision is available.  */
extern int __pthread_force_elision;
libc_hidden_proto (__pthread_force_elision)

extern void __lll_elision_init (void) attribute_hidden;
extern int __lll_clocklock_elision (int *futex, short *adapt_count,
                                    clockid_t clockid,
				    const struct __timespec64 *timeout,
				    int private);
libc_hidden_proto (__lll_clocklock_elision)

extern int __lll_lock_elision (int *futex, short *adapt_count, int private);
libc_hidden_proto (__lll_lock_elision)

# if ELISION_UNLOCK_NEEDS_ADAPT_COUNT
extern int __lll_unlock_elision (int *lock, short *adapt_count, int private);
# else
extern int __lll_unlock_elision (int *lock, int private);
# endif
libc_hidden_proto (__lll_unlock_elision)

extern int __lll_trylock_elision (int *lock, short *adapt_count);
libc_hidden_proto (__lll_trylock_elision)

# define lll_clocklock_elision(futex, adapt_count, clockid, timeout, private) \
  __lll_clocklock_elision (&(futex), &(adapt_count), clockid, timeout, private)
# define lll_lock_elision(futex, adapt_count, private)		\
  __lll_lock_elision (&(futex), &(adapt_count), private)
# define lll_trylock_elision(futex, adapt_count)	\
  __lll_trylock_elision (&(futex), &(adapt_count))
# if ELISION_UNLOCK_NEEDS_ADAPT_COUNT
#  define lll_unlock_elision(futex, adapt_count, private)	\
  __lll_unlock_elision (&(futex), &(adapt_count), private)
# else
#  define lll_unlock_elision(futex, adapt_count, private)	\
  __lll_unlock_elision (&(futex), private)
# endif

/* Automatically enable elision for existing user lock kinds.  */
# define FORCE_ELISION(m, s)                                            \
  if (__pthread_force_elision)                                          \
    {                                                                   \
      /* See concurrency notes regarding __kind in                      \
         struct __pthread_mutex_s in                                    \
         sysdeps/nptl/bits/thread-shared-types.h.                       \
                                                                        \
         There are the following cases for the kind of a mutex          \
         (The mask PTHREAD_MUTEX_ELISION_FLAGS_NP covers the flags      \
         PTHREAD_MUTEX_ELISION_NP and PTHREAD_MUTEX_NO_ELISION_NP where \
         only one of both flags can be set):                            \
         - both flags are not set:                                      \
         This is the first lock operation for this mutex.  Enable       \
         elision as it is not enabled so far.                           \
         Note: It can happen that multiple threads are calling e.g.     \
         pthread_mutex_lock at the same time as the first lock          \
         operation for this mutex.  Then elision is enabled for this    \
         mutex by multiple threads.  Storing with relaxed MO is enough  \
         as all threads will store the same new value for the kind of   \
         the mutex.  But we have to ensure that we always use the       \
         elision path regardless if this thread has enabled elision or  \
         another one.                                                   \
                                                                        \
         - PTHREAD_MUTEX_ELISION_NP flag is set:                        \
         Elision was already enabled for this mutex by a previous lock  \
         operation.  See case above.  Just use the elision path.        \
                                                                        \
         - PTHREAD_MUTEX_NO_ELISION_NP flag is set:                     \
         Elision was explicitly disabled by pthread_mutexattr_settype.  \
         Do not use the elision path.                                   \
         Note: The flag PTHREAD_MUTEX_NO_ELISION_NP will never be       \
         changed after mutex initialization.  */                        \
      int mutex_kind = atomic_load_relaxed (&((m)->__data.__kind));     \
      if ((mutex_kind & PTHREAD_MUTEX_ELISION_FLAGS_NP) == 0)           \
        {                                                               \
          mutex_kind |= PTHREAD_MUTEX_ELISION_NP;                       \
          atomic_store_relaxed (&((m)->__data.__kind), mutex_kind);     \
        }                                                               \
      if ((mutex_kind & PTHREAD_MUTEX_ELISION_NP) != 0)                 \
        {                                                               \
          s;                                                            \
        }                                                               \
    }

#else /* !ENABLE_ELISION_SUPPORT */

# define lll_clocklock_elision(futex, adapt_count, clockid, abstime, private) \
  __futex_clocklock64 (&(futex), clockid, abstime, private)
# define lll_lock_elision(lock, try_lock, private)	\
  ({ lll_lock (lock, private); 0; })
# define lll_trylock_elision(a,t) lll_trylock(a)
# define lll_unlock_elision(a,b,c) ({ lll_unlock (a,c); 0; })
# define FORCE_ELISION(m, s)

#endif /* !ENABLE_ELISION_SUPPORT */

#endif	/* lowlevellock.h */
