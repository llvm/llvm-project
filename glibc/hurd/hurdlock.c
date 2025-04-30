/* Hurd helpers for lowlevellocks.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include "hurdlock.h"
#include <hurd.h>
#include <hurd/hurd.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

/* Convert an absolute timeout in nanoseconds to a relative
   timeout in milliseconds.  */
static inline int __attribute__ ((gnu_inline))
compute_reltime (const struct timespec *abstime, clockid_t clk)
{
  struct timespec ts;
  __clock_gettime (clk, &ts);

  ts.tv_sec = abstime->tv_sec - ts.tv_sec;
  ts.tv_nsec = abstime->tv_nsec - ts.tv_nsec;

  if (ts.tv_nsec < 0)
    {
      --ts.tv_sec;
      ts.tv_nsec += 1000000000;
    }

  return ts.tv_sec < 0 ? -1 : (int)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

int
__lll_abstimed_wait (void *ptr, int val,
  const struct timespec *tsp, int flags, int clk)
{
  if (clk != CLOCK_REALTIME)
    return EINVAL;

  int mlsec = compute_reltime (tsp, clk);
  return mlsec < 0 ? KERN_TIMEDOUT : __lll_timed_wait (ptr, val, mlsec, flags);
}

int
__lll_abstimed_wait_intr (void *ptr, int val,
  const struct timespec *tsp, int flags, int clk)
{
  if (clk != CLOCK_REALTIME)
    return EINVAL;

  int mlsec = compute_reltime (tsp, clk);
  return mlsec < 0 ? KERN_TIMEDOUT : __lll_timed_wait_intr (ptr, val, mlsec, flags);
}

int
__lll_abstimed_xwait (void *ptr, int lo, int hi,
  const struct timespec *tsp, int flags, int clk)
{
  if (clk != CLOCK_REALTIME)
    return EINVAL;

  int mlsec = compute_reltime (tsp, clk);
  return mlsec < 0 ? KERN_TIMEDOUT : __lll_timed_xwait (ptr, lo, hi, mlsec,
	                                              flags);
}

int
__lll_abstimed_lock (void *ptr,
  const struct timespec *tsp, int flags, int clk)
{
  if (clk != CLOCK_REALTIME)
    return EINVAL;

  if (__lll_trylock (ptr) == 0)
    return 0;

  while (1)
    {
      if (atomic_exchange_acq ((int *)ptr, 2) == 0)
        return 0;
      else if (! valid_nanoseconds (tsp->tv_nsec))
        return EINVAL;

      int mlsec = compute_reltime (tsp, clk);
      if (mlsec < 0 || __lll_timed_wait (ptr, 2, mlsec, flags) == KERN_TIMEDOUT)
        return ETIMEDOUT;
    }
}

/* Robust locks.  */

/* Test if a given process id is still valid.  */
static inline int
valid_pid (int pid)
{
  task_t task = __pid2task (pid);
  if (task == MACH_PORT_NULL)
    return 0;

  __mach_port_deallocate (__mach_task_self (), task);
  return 1;
}

/* Robust locks have currently no support from the kernel; they
   are simply implemented with periodic polling. When sleeping, the
   maximum blocking time is determined by this constant.  */
#define MAX_WAIT_TIME   1500

int
__lll_robust_lock (void *ptr, int flags)
{
  int *iptr = (int *)ptr;
  int id = __getpid ();
  int wait_time = 25;
  unsigned int val;

  /* Try to set the lock word to our PID if it's clear. Otherwise,
     mark it as having waiters.  */
  while (1)
    {
      val = *iptr;
      if (!val && atomic_compare_and_exchange_bool_acq (iptr, id, 0) == 0)
        return 0;
      else if (atomic_compare_and_exchange_bool_acq (iptr,
               val | LLL_WAITERS, val) == 0)
        break;
    }

  for (id |= LLL_WAITERS ; ; )
    {
      val = *iptr;
      if (!val && atomic_compare_and_exchange_bool_acq (iptr, id, 0) == 0)
        return 0;
      else if (val && !valid_pid (val & LLL_OWNER_MASK))
        {
          if (atomic_compare_and_exchange_bool_acq (iptr, id, val) == 0)
            return EOWNERDEAD;
        }
      else
        {
          __lll_timed_wait (iptr, val, wait_time, flags);
          if (wait_time < MAX_WAIT_TIME)
            wait_time <<= 1;
        }
    }
}

int
__lll_robust_abstimed_lock (void *ptr,
  const struct timespec *tsp, int flags, int clk)
{
  int *iptr = (int *)ptr;
  int id = __getpid ();
  int wait_time = 25;
  unsigned int val;

  if (clk != CLOCK_REALTIME)
    return EINVAL;

  while (1)
    {
      val = *iptr;
      if (!val && atomic_compare_and_exchange_bool_acq (iptr, id, 0) == 0)
        return 0;
      else if (atomic_compare_and_exchange_bool_acq (iptr,
          val | LLL_WAITERS, val) == 0)
        break;
    }

  for (id |= LLL_WAITERS ; ; )
    {
      val = *iptr;
      if (!val && atomic_compare_and_exchange_bool_acq (iptr, id, 0) == 0)
        return 0;
      else if (val && !valid_pid (val & LLL_OWNER_MASK))
        {
          if (atomic_compare_and_exchange_bool_acq (iptr, id, val) == 0)
            return EOWNERDEAD;
        }
      else
        {
          int mlsec = compute_reltime (tsp, clk);
          if (mlsec < 0)
            return ETIMEDOUT;
          else if (mlsec > wait_time)
            mlsec = wait_time;

          int res = __lll_timed_wait (iptr, val, mlsec, flags);
          if (res == KERN_TIMEDOUT)
            return ETIMEDOUT;
          else if (wait_time < MAX_WAIT_TIME)
            wait_time <<= 1;
        }
    }
}

int
__lll_robust_trylock (void *ptr)
{
  int *iptr = (int *)ptr;
  int id = __getpid ();
  unsigned int val = *iptr;

  if (!val)
    {
      if (atomic_compare_and_exchange_bool_acq (iptr, id, 0) == 0)
        return 0;
    }
  else if (!valid_pid (val & LLL_OWNER_MASK)
           && atomic_compare_and_exchange_bool_acq (iptr, id, val) == 0)
    return EOWNERDEAD;

  return EBUSY;
}

void
__lll_robust_unlock (void *ptr, int flags)
{
  unsigned int val = atomic_load_relaxed ((unsigned int *)ptr);
  while (1)
    {
      if (val & LLL_WAITERS)
        {
          __lll_set_wake (ptr, 0, flags);
          break;
        }
      else if (atomic_compare_exchange_weak_release ((unsigned int *)ptr, &val, 0))
        break;
    }
}
