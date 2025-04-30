/* Low-level lock implementation.  Mach gsync-based version.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _MACH_LOWLEVELLOCK_H
#define _MACH_LOWLEVELLOCK_H   1

#include <mach/gnumach.h>
#include <atomic.h>

/* Gsync flags.  */
#ifndef GSYNC_SHARED
# define GSYNC_SHARED      0x01
# define GSYNC_QUAD        0x02
# define GSYNC_TIMED       0x04
# define GSYNC_BROADCAST   0x08
# define GSYNC_MUTATE      0x10
#endif

/* Static initializer for low-level locks.  */
#define LLL_LOCK_INITIALIZER   0

#define LLL_PRIVATE        0
#define LLL_SHARED         SYNC_SHARED

/* Interruptible version of __gsync_wait.  */
extern kern_return_t __gsync_wait_intr
(
	mach_port_t task,
	vm_offset_t addr,
	unsigned val1,
	unsigned val2,
	natural_t msec,
	int flags
);

/* Wait on address PTR, without blocking if its contents
 * are different from VAL.  */
#define __lll_wait(ptr, val, flags)   \
  __gsync_wait (__mach_task_self (),   \
    (vm_offset_t)(ptr), (val), 0, 0, (flags))
#define lll_wait(var, val, flags) \
  __lll_wait (&(var), val, flags)

/* Interruptible version.  */
#define __lll_wait_intr(ptr, val, flags)   \
  __gsync_wait_intr (__mach_task_self (),   \
    (vm_offset_t)(ptr), (val), 0, 0, (flags))
#define lll_wait_intr(var, val, flags) \
  __lll_wait_intr ((&var), val, flags)

/* Wake one or more threads waiting on address PTR.  */
#define __lll_wake(ptr, flags)   \
  __gsync_wake (__mach_task_self (), (vm_offset_t)(ptr), 0, (flags))
#define lll_wake(var, flags) \
  __lll_wake (&(var), flags)

/* Acquire the lock at PTR.  */
#define __lll_lock(ptr, flags)   \
  ({   \
     int *__iptr = (int *)(ptr);   \
     int __flags = (flags);   \
     if (*__iptr != 0   \
         || atomic_compare_and_exchange_bool_acq (__iptr, 1, 0) != 0)   \
       while (1)   \
         {   \
           if (atomic_exchange_acq (__iptr, 2) == 0)   \
             break;   \
           __lll_wait (__iptr, 2, __flags);   \
         }   \
     (void)0;   \
   })
#define lll_lock(var, flags) \
  __lll_lock (&(var), flags)

/* Try to acquire the lock at PTR, without blocking.
   Evaluates to zero on success.  */
#define __lll_trylock(ptr)   \
  ({   \
     int *__iptr = (int *)(ptr);   \
     *__iptr == 0   \
       && atomic_compare_and_exchange_bool_acq (__iptr, 1, 0) == 0 ? 0 : -1;   \
   })
#define lll_trylock(var) \
  __lll_trylock (&(var))

/* Release the lock at PTR.  */
#define __lll_unlock(ptr, flags)   \
  ({   \
     int *__iptr = (int *)(ptr);   \
     if (atomic_exchange_rel (__iptr, 0) == 2)   \
       __lll_wake (__iptr, (flags));   \
     (void)0;   \
   })
#define lll_unlock(var, flags) \
  __lll_unlock (&(var), flags)

#endif
