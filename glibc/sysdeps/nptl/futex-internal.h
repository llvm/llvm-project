/* futex operations for glibc-internal use.  Stub version; do not include
   this file directly.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef STUB_FUTEX_INTERNAL_H
#define STUB_FUTEX_INTERNAL_H

#include <sys/time.h>
#include <stdio.h>
#include <stdbool.h>
#include <lowlevellock-futex.h>
#include <libc-diag.h>

/* This file defines futex operations used internally in glibc.  A futex
   consists of the so-called futex word in userspace, which is of type
   unsigned int and represents an application-specific condition, and kernel
   state associated with this particular futex word (e.g., wait queues).  The
   futex operations we provide are wrappers for the futex syscalls and add
   glibc-specific error checking of the syscall return value.  We abort on
   error codes that are caused by bugs in glibc or in the calling application,
   or when an error code is not known.  We return error codes that can arise
   in correct executions to the caller.  Each operation calls out exactly the
   return values that callers need to handle.

   The private flag must be either FUTEX_PRIVATE or FUTEX_SHARED.
   FUTEX_PRIVATE is always supported, and the implementation can internally
   use FUTEX_SHARED when FUTEX_PRIVATE is requested.  FUTEX_SHARED is not
   necessarily supported (use futex_supports_pshared to detect this).

   We expect callers to only use these operations if futexes and the
   specific futex operations being used are supported (e.g., FUTEX_SHARED).

   Given that waking other threads waiting on a futex involves concurrent
   accesses to the futex word, you must use atomic operations to access the
   futex word.

   Both absolute and relative timeouts can be used.  An absolute timeout
   expires when the given specific point in time on the specified clock
   passes, or when it already has passed.  A relative timeout expires when
   the given duration of time on the CLOCK_MONOTONIC clock passes.

   Due to POSIX requirements on when synchronization data structures such
   as mutexes or semaphores can be destroyed and due to the futex design
   having separate fast/slow paths for wake-ups, we need to consider that
   futex_wake calls might effectively target a data structure that has been
   destroyed and reused for another object, or unmapped; thus, some
   errors or spurious wake-ups can happen in correct executions that would
   not be possible in a program using just a single futex whose lifetime
   does not end before the program terminates.  For background, see:
   https://sourceware.org/ml/libc-alpha/2014-04/msg00075.html
   https://lkml.org/lkml/2014/11/27/472  */

/* Defined this way for interoperability with lowlevellock.
   FUTEX_PRIVATE must be zero because the initializers for pthread_mutex_t,
   pthread_rwlock_t, and pthread_cond_t initialize the respective field of
   those structures to zero, and we want FUTEX_PRIVATE to be the default.  */
#define FUTEX_PRIVATE LLL_PRIVATE
#define FUTEX_SHARED  LLL_SHARED
#if FUTEX_PRIVATE != 0
# error FUTEX_PRIVATE must be equal to 0
#endif

#ifndef __NR_futex_time64
# define __NR_futex_time64 __NR_futex
#endif

/* Calls __libc_fatal with an error message.  Convenience function for
   concrete implementations of the futex interface.  */
static __always_inline __attribute__ ((__noreturn__)) void
futex_fatal_error (void)
{
  __libc_fatal ("The futex facility returned an unexpected error code.\n");
}


/* The Linux kernel treats provides absolute timeouts based on the
   CLOCK_REALTIME clock and relative timeouts measured against the
   CLOCK_MONOTONIC clock.

   We expect a Linux kernel version of 2.6.22 or more recent (since this
   version, EINTR is not returned on spurious wake-ups anymore).  */

/* Returns EINVAL if PSHARED is neither PTHREAD_PROCESS_PRIVATE nor
   PTHREAD_PROCESS_SHARED; otherwise, returns 0 if PSHARED is supported, and
   ENOTSUP if not.  */
static __always_inline int
futex_supports_pshared (int pshared)
{
  if (__glibc_likely (pshared == PTHREAD_PROCESS_PRIVATE))
    return 0;
  else if (pshared == PTHREAD_PROCESS_SHARED)
    return 0;
  else
    return EINVAL;
}

/* Atomically wrt other futex operations on the same futex, this blocks iff
   the value *FUTEX_WORD matches the expected value.  This is
   semantically equivalent to:
     l = <get lock associated with futex> (FUTEX_WORD);
     wait_flag = <get wait_flag associated with futex> (FUTEX_WORD);
     lock (l);
     val = atomic_load_relaxed (FUTEX_WORD);
     if (val != expected) { unlock (l); return EAGAIN; }
     atomic_store_relaxed (wait_flag, true);
     unlock (l);
     // Now block; can time out in futex_time_wait (see below)
     while (atomic_load_relaxed(wait_flag) && !<spurious wake-up>);

   Note that no guarantee of a happens-before relation between a woken
   futex_wait and a futex_wake is documented; however, this does not matter
   in practice because we have to consider spurious wake-ups (see below),
   and thus would not be able to reliably reason about which futex_wake woke
   us.

   Returns 0 if woken by a futex operation or spuriously.  (Note that due to
   the POSIX requirements mentioned above, we need to conservatively assume
   that unrelated futex_wake operations could wake this futex; it is easiest
   to just be prepared for spurious wake-ups.)
   Returns EAGAIN if the futex word did not match the expected value.
   Returns EINTR if waiting was interrupted by a signal.

   Note that some previous code in glibc assumed the underlying futex
   operation (e.g., syscall) to start with or include the equivalent of a
   seq_cst fence; this allows one to avoid an explicit seq_cst fence before
   a futex_wait call when synchronizing similar to Dekker synchronization.
   However, we make no such guarantee here.  */
static __always_inline int
futex_wait (unsigned int *futex_word, unsigned int expected, int private)
{
  int err = lll_futex_timed_wait (futex_word, expected, NULL, private);
  switch (err)
    {
    case 0:
    case -EAGAIN:
    case -EINTR:
      return -err;

    case -ETIMEDOUT: /* Cannot have happened as we provided no timeout.  */
    case -EFAULT: /* Must have been caused by a glibc or application bug.  */
    case -EINVAL: /* Either due to wrong alignment or due to the timeout not
		     being normalized.  Must have been caused by a glibc or
		     application bug.  */
    case -ENOSYS: /* Must have been caused by a glibc bug.  */
    /* No other errors are documented at this time.  */
    default:
      futex_fatal_error ();
    }
}

/* Like futex_wait but does not provide any indication why we stopped waiting.
   Thus, when this function returns, you have to always check FUTEX_WORD to
   determine whether you need to continue waiting, and you cannot detect
   whether the waiting was interrupted by a signal.  Example use:
     while (atomic_load_relaxed (&futex_word) == 23)
       futex_wait_simple (&futex_word, 23, FUTEX_PRIVATE);
   This is common enough to make providing this wrapper worthwhile.  */
static __always_inline void
futex_wait_simple (unsigned int *futex_word, unsigned int expected,
		   int private)
{
  ignore_value (futex_wait (futex_word, expected, private));
}

/* Check whether the specified clockid is supported by
   futex_abstimed_wait and futex_abstimed_wait_cancelable.  */
static __always_inline int
futex_abstimed_supported_clockid (clockid_t clockid)
{
  return lll_futex_supported_clockid (clockid);
}

/* Atomically wrt other futex operations on the same futex, this unblocks the
   specified number of processes, or all processes blocked on this futex if
   there are fewer than the specified number.  Semantically, this is
   equivalent to:
     l = <get lock associated with futex> (FUTEX_WORD);
     lock (l);
     for (res = 0; PROCESSES_TO_WAKE > 0; PROCESSES_TO_WAKE--, res++) {
       if (<no process blocked on futex>) break;
       wf = <get wait_flag of a process blocked on futex> (FUTEX_WORD);
       // No happens-before guarantee with woken futex_wait (see above)
       atomic_store_relaxed (wf, 0);
     }
     return res;

   Note that we need to support futex_wake calls to past futexes whose memory
   has potentially been reused due to POSIX' requirements on synchronization
   object destruction (see above); therefore, we must not report or abort
   on most errors.  */
static __always_inline void
futex_wake (unsigned int* futex_word, int processes_to_wake, int private)
{
  int res = lll_futex_wake (futex_word, processes_to_wake, private);
  /* No error.  Ignore the number of woken processes.  */
  if (res >= 0)
    return;
  switch (res)
    {
    case -EFAULT: /* Could have happened due to memory reuse.  */
    case -EINVAL: /* Could be either due to incorrect alignment (a bug in
		     glibc or in the application) or due to memory being
		     reused for a PI futex.  We cannot distinguish between the
		     two causes, and one of them is correct use, so we do not
		     act in this case.  */
      return;
    case -ENOSYS: /* Must have been caused by a glibc bug.  */
    /* No other errors are documented at this time.  */
    default:
      futex_fatal_error ();
    }
}

/* The operation checks the value of the futex, if the value is 0, then
   it is atomically set to the caller's thread ID.  If the futex value is
   nonzero, it is atomically sets the FUTEX_WAITERS bit, which signals wrt
   other futex owner that it cannot unlock the futex in user space by
   atomically by setting its value to 0.

   If more than one wait operations is issued, the enqueueing of the waiters
   are done in descending priority order.

   The ABSTIME arguments provides an absolute timeout (measured against the
   CLOCK_REALTIME clock).  If TIMEOUT is NULL, the operation will block
   indefinitely.

   Returns:

     - 0 if woken by a PI unlock operation or spuriously.
     - EAGAIN if the futex owner thread ID is about to exit, but has not yet
       handled the state cleanup.
     - EDEADLK if the futex is already locked by the caller.
     - ESRCH if the thread ID int he futex does not exist.
     - EINVAL is the state is corrupted or if there is a waiter on the
       futex.
     - ETIMEDOUT if the ABSTIME expires.
*/
static __always_inline int
futex_lock_pi64 (int *futex_word, const struct __timespec64 *abstime,
                 int private)
{
  int err;
#ifdef __ASSUME_TIME64_SYSCALLS
  err = INTERNAL_SYSCALL_CALL (futex_time64, futex_word,
			       __lll_private_flag (FUTEX_LOCK_PI, private), 0,
			       abstime);
#else
  bool need_time64 = abstime != NULL && !in_time_t_range (abstime->tv_sec);
  if (need_time64)
    {
      err = INTERNAL_SYSCALL_CALL (futex_time64, futex_word,
				   __lll_private_flag (FUTEX_LOCK_PI, private),
				   0, abstime);
      if (err == -ENOSYS)
	err = -EOVERFLOW;
    }
  else
    {
      struct timespec ts32;
      if (abstime != NULL)
        ts32 = valid_timespec64_to_timespec (*abstime);

      err = INTERNAL_SYSCALL_CALL (futex, futex_word, __lll_private_flag
                                   (FUTEX_LOCK_PI, private), 0,
                                   abstime != NULL ? &ts32 : NULL);
    }
#endif
  switch (err)
    {
    case 0:
    case -EAGAIN:
    case -EINTR:
    case -ETIMEDOUT:
    case -ESRCH:
    case -EDEADLK:
    case -EINVAL: /* This indicates either state corruption or that the kernel
		     found a waiter on futex address which is waiting via
		     FUTEX_WAIT or FUTEX_WAIT_BITSET.  This is reported on
		     some futex_lock_pi usage (pthread_mutex_timedlock for
		     instance).  */
      return -err;

    case -EFAULT: /* Must have been caused by a glibc or application bug.  */
    case -ENOSYS: /* Must have been caused by a glibc bug.  */
    /* No other errors are documented at this time.  */
    default:
      futex_fatal_error ();
    }
}

/* Wakes the top priority waiter that called a futex_lock_pi operation on
   the futex.

   Returns the same values as futex_lock_pi under those same conditions;
   additionally, returns EPERM when the caller is not allowed to attach
   itself to the futex.  */
static __always_inline int
futex_unlock_pi (unsigned int *futex_word, int private)
{
  int err = lll_futex_timed_unlock_pi (futex_word, private);
  switch (err)
    {
    case 0:
    case -EAGAIN:
    case -EINTR:
    case -ETIMEDOUT:
    case -ESRCH:
    case -EDEADLK:
    case -ENOSYS:
    case -EPERM:  /*  The caller is not allowed to attach itself to the futex.
		      Used to check if PI futexes are supported by the
		      kernel.  */
      return -err;

    case -EINVAL: /* Either due to wrong alignment or due to the timeout not
		     being normalized.  Must have been caused by a glibc or
		     application bug.  */
    case -EFAULT: /* Must have been caused by a glibc or application bug.  */
    /* No other errors are documented at this time.  */
    default:
      futex_fatal_error ();
    }
}

/* Like futex_wait, but will eventually time out (i.e., stop being blocked)
   after the duration of time provided (i.e., ABSTIME) has passed using the
   clock specified by CLOCKID (currently only CLOCK_REALTIME and
   CLOCK_MONOTONIC, the ones support by lll_futex_supported_clockid). ABSTIME
   can also equal NULL, in which case this function behaves equivalent to
   futex_wait.

   Returns the same values as futex_wait under those same conditions;
   additionally, returns ETIMEDOUT if the timeout expired.

   The call acts as a cancellation entrypoint.  */
int
__futex_abstimed_wait_cancelable64 (unsigned int* futex_word,
                                    unsigned int expected, clockid_t clockid,
                                    const struct __timespec64* abstime,
                                    int private);
libc_hidden_proto (__futex_abstimed_wait_cancelable64);

int
__futex_abstimed_wait64 (unsigned int* futex_word, unsigned int expected,
                         clockid_t clockid,
                         const struct __timespec64* abstime,
                         int private);
libc_hidden_proto (__futex_abstimed_wait64);


static __always_inline int
__futex_clocklock64 (int *futex, clockid_t clockid,
                     const struct __timespec64 *abstime, int private)
{
  if (__glibc_unlikely (atomic_compare_and_exchange_bool_acq (futex, 1, 0)))
    {
      while (atomic_exchange_acq (futex, 2) != 0)
        {
	  int err = 0;
          err = __futex_abstimed_wait64 ((unsigned int *) futex, 2, clockid,
					 abstime, private);
          if (err == EINVAL || err == ETIMEDOUT || err == EOVERFLOW)
            return err;
        }
    }
  return 0;
}

#endif  /* futex-internal.h */
