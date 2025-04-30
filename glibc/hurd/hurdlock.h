/* Low-level lock implementation.  High-level Hurd helpers.
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

#ifndef _HURD_LOCK_H
#define _HURD_LOCK_H   1

#include <mach/lowlevellock.h>

struct timespec;

/* Flags for robust locks.  */
#define LLL_WAITERS      (1U << 31)
#define LLL_DEAD_OWNER   (1U << 30)

#define LLL_OWNER_MASK   ~(LLL_WAITERS | LLL_DEAD_OWNER)

/* Wait on 64-bit address PTR, without blocking if its contents
   are different from the pair <LO, HI>.  */
#define __lll_xwait(ptr, lo, hi, flags) \
  __gsync_wait (__mach_task_self (), \
    (vm_offset_t)ptr, lo, hi, 0, flags | GSYNC_QUAD)

/* Same as '__lll_wait', but only block for MLSEC milliseconds.  */
#define __lll_timed_wait(ptr, val, mlsec, flags) \
  __gsync_wait (__mach_task_self (), \
    (vm_offset_t)ptr, val, 0, mlsec, flags | GSYNC_TIMED)

/* Interruptible version.  */
#define __lll_timed_wait_intr(ptr, val, mlsec, flags) \
  __gsync_wait_intr (__mach_task_self (), \
    (vm_offset_t)ptr, val, 0, mlsec, flags | GSYNC_TIMED)

/* Same as '__lll_xwait', but only block for MLSEC milliseconds.  */
#define __lll_timed_xwait(ptr, lo, hi, mlsec, flags) \
  __gsync_wait (__mach_task_self (), (vm_offset_t)ptr, \
    lo, hi, mlsec, flags | GSYNC_TIMED | GSYNC_QUAD)

/* Same as '__lll_wait', but only block until TSP elapses,
   using clock CLK.  */
extern int __lll_abstimed_wait (void *__ptr, int __val,
  const struct timespec *__tsp, int __flags, int __clk);

/* Interruptible version.  */
extern int __lll_abstimed_wait_intr (void *__ptr, int __val,
  const struct timespec *__tsp, int __flags, int __clk);

/* Same as 'lll_xwait', but only block until TSP elapses,
   using clock CLK.  */
extern int __lll_abstimed_xwait (void *__ptr, int __lo, int __hi,
  const struct timespec *__tsp, int __flags, int __clk);

/* Same as 'lll_lock', but return with an error if TSP elapses,
   using clock CLK.  */
extern int __lll_abstimed_lock (void *__ptr,
  const struct timespec *__tsp, int __flags, int __clk);

/* Acquire the lock at PTR, but return with an error if
   the process containing the owner thread dies.  */
extern int __lll_robust_lock (void *__ptr, int __flags);
#define lll_robust_lock(var, flags) \
  __lll_robust_lock (&(var), flags)

/* Same as '__lll_robust_lock', but only block until TSP
   elapses, using clock CLK.  */
extern int __lll_robust_abstimed_lock (void *__ptr,
  const struct timespec *__tsp, int __flags, int __clk);

/* Same as '__lll_robust_lock', but return with an error
   if the lock cannot be acquired without blocking.  */
extern int __lll_robust_trylock (void *__ptr);
#define lll_robust_trylock(var) \
  __lll_robust_trylock (&(var))

/* Wake one or more threads waiting on address PTR,
   setting its value to VAL before doing so.  */
#define __lll_set_wake(ptr, val, flags) \
  __gsync_wake (__mach_task_self (), \
    (vm_offset_t)ptr, val, flags | GSYNC_MUTATE)

/* Release the robust lock at PTR.  */
extern void __lll_robust_unlock (void *__ptr, int __flags);
#define lll_robust_unlock(var, flags) \
  __lll_robust_unlock (&(var), flags)

/* Rearrange threads waiting on address SRC to instead wait on
   DST, waking one of them if WAIT_ONE is non-zero.  */
#define __lll_requeue(src, dst, wake_one, flags) \
  __gsync_requeue (__mach_task_self (), (vm_offset_t)src, \
    (vm_offset_t)dst, (boolean_t)wake_one, flags)

/* The following are hacks that allow us to simulate optional
   parameters in C, to avoid having to pass the clock id for
   every one of these calls, defaulting to CLOCK_REALTIME if
   no argument is passed.  */

#define lll_abstimed_wait(var, val, tsp, flags, ...)   \
  ({   \
     const clockid_t __clk[] = { CLOCK_REALTIME, ##__VA_ARGS__ };   \
     __lll_abstimed_wait (&(var), (val), (tsp), (flags),   \
       __clk[sizeof (__clk) / sizeof (__clk[0]) - 1]);   \
   })

#define lll_abstimed_wait_intr(var, val, tsp, flags, ...)   \
  ({   \
     const clockid_t __clk[] = { CLOCK_REALTIME, ##__VA_ARGS__ };   \
     __lll_abstimed_wait_intr (&(var), (val), (tsp), (flags),   \
       __clk[sizeof (__clk) / sizeof (__clk[0]) - 1]);   \
   })

#define lll_abstimed_xwait(var, lo, hi, tsp, flags, ...)   \
  ({   \
     const clockid_t __clk[] = { CLOCK_REALTIME, ##__VA_ARGS__ };   \
     __lll_abstimed_xwait (&(var), (lo), (hi), (tsp), (flags),   \
       __clk[sizeof (__clk) / sizeof (__clk[0]) - 1]);   \
   })

#define lll_abstimed_lock(var, tsp, flags, ...)   \
  ({   \
     const clockid_t __clk[] = { CLOCK_REALTIME, ##__VA_ARGS__ };   \
     __lll_abstimed_lock (&(var), (tsp), (flags),   \
       __clk[sizeof (__clk) / sizeof (__clk[0]) - 1]);   \
   })

#define lll_robust_abstimed_lock(var, tsp, flags, ...)   \
  ({   \
     const clockid_t __clk[] = { CLOCK_REALTIME, ##__VA_ARGS__ };   \
     __lll_robust_abstimed_lock (&(var), (tsp), (flags),   \
       __clk[sizeof (__clk) / sizeof (__clk[0]) - 1]);   \
   })


#endif
