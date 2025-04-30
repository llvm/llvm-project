/* futex helper functions for glibc-internal use.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <sysdep.h>
#include <time.h>
#include <futex-internal.h>
#include <kernel-features.h>
#include <_ns_unmigratable.h>

#ifndef __ASSUME_TIME64_SYSCALLS
static int
__futex_abstimed_wait_common32 (unsigned int* futex_word,
                                unsigned int expected, int op,
                                const struct __timespec64* abstime,
                                int private, bool cancel)
{
  struct timespec ts32, *pts32 = NULL;
  if (abstime != NULL)
    {
      ts32 = valid_timespec64_to_timespec (*abstime);
      pts32 = &ts32;
    }

  if (cancel)
    return INTERNAL_SYSCALL_CANCEL (futex, futex_word, op, expected,
                                    pts32, NULL /* Unused.  */,
                                    FUTEX_BITSET_MATCH_ANY);
  else
    return INTERNAL_SYSCALL_CALL (futex, futex_word, op, expected,
                                  pts32, NULL /* Unused.  */,
                                  FUTEX_BITSET_MATCH_ANY);
}
#endif /* ! __ASSUME_TIME64_SYSCALLS */

static int
__futex_abstimed_wait_common64 (unsigned int* futex_word,
                                unsigned int expected, int op,
                                const struct __timespec64* abstime,
                                int private, bool cancel)
{
  __try_to_mark_as_unmigratable(futex_word);
  if (cancel)
    return INTERNAL_SYSCALL_CANCEL (futex_time64, futex_word, op, expected,
				    abstime, NULL /* Unused.  */,
				    FUTEX_BITSET_MATCH_ANY);
  else
    return INTERNAL_SYSCALL_CALL (futex_time64, futex_word, op, expected,
				  abstime, NULL /* Ununsed.  */,
				  FUTEX_BITSET_MATCH_ANY);
}

static int
__futex_abstimed_wait_common (unsigned int* futex_word,
                              unsigned int expected, clockid_t clockid,
                              const struct __timespec64* abstime,
                              int private, bool cancel)
{
  int err;
  unsigned int clockbit;

  /* Work around the fact that the kernel rejects negative timeout values
     despite them being valid.  */
  if (__glibc_unlikely ((abstime != NULL) && (abstime->tv_sec < 0)))
    return ETIMEDOUT;

  if (! lll_futex_supported_clockid (clockid))
    return EINVAL;

  clockbit = (clockid == CLOCK_REALTIME) ? FUTEX_CLOCK_REALTIME : 0;
  int op = __lll_private_flag (FUTEX_WAIT_BITSET | clockbit, private);

#ifdef __ASSUME_TIME64_SYSCALLS
  err = __futex_abstimed_wait_common64 (futex_word, expected, op, abstime,
					private, cancel);
#else
  bool need_time64 = abstime != NULL && !in_time_t_range (abstime->tv_sec);
  if (need_time64)
    {
      err = __futex_abstimed_wait_common64 (futex_word, expected, op, abstime,
					    private, cancel);
      if (err == -ENOSYS)
	err = -EOVERFLOW;
    }
  else
    err = __futex_abstimed_wait_common32 (futex_word, expected, op, abstime,
                                          private, cancel);
#endif

  switch (err)
    {
    case 0:
    case -EAGAIN:
    case -EINTR:
    case -ETIMEDOUT:
    case -EINVAL:
    case -EOVERFLOW:  /* Passed absolute timeout uses 64 bit time_t type, but
                         underlying kernel does not support 64 bit time_t futex
                         syscalls.  */
      return -err;

    case -EFAULT: /* Must have been caused by a glibc or application bug.  */
    case -ENOSYS: /* Must have been caused by a glibc bug.  */
    /* No other errors are documented at this time.  */
    default:
      futex_fatal_error ();
    }
}

int
__futex_abstimed_wait64 (unsigned int* futex_word, unsigned int expected,
                         clockid_t clockid,
                         const struct __timespec64* abstime, int private)
{
  return __futex_abstimed_wait_common (futex_word, expected, clockid,
                                       abstime, private, false);
}
libc_hidden_def (__futex_abstimed_wait64)

int
__futex_abstimed_wait_cancelable64 (unsigned int* futex_word,
                                    unsigned int expected, clockid_t clockid,
                                    const struct __timespec64* abstime,
                                    int private)
{
  return __futex_abstimed_wait_common (futex_word, expected, clockid,
                                       abstime, private, true);
}
libc_hidden_def (__futex_abstimed_wait_cancelable64)
