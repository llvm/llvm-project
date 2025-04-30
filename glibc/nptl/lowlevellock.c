/* low level locking for pthread library.  Generic futex-using version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Paul Mackerras <paulus@au.ibm.com>, 2003.

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

#include <sysdep.h>
#include <futex-internal.h>
#include <atomic.h>
#include <stap-probe.h>

int __ns_main_app_started;

void
__lll_lock_wait_private (int *futex)
{
  if (atomic_load_relaxed (futex) == 2)
    goto futex;

  while (atomic_exchange_acquire (futex, 2) != 0)
    {
    futex:
      LIBC_PROBE (lll_lock_wait_private, 1, futex);
      futex_wait ((unsigned int *) futex, 2, LLL_PRIVATE); /* Wait if *futex == 2.  */
    }
}
libc_hidden_def (__lll_lock_wait_private)

void
__lll_lock_wait (int *futex, int private)
{
  if (atomic_load_relaxed (futex) == 2)
    goto futex;

  while (atomic_exchange_acquire (futex, 2) != 0)
    {
    futex:
      LIBC_PROBE (lll_lock_wait, 1, futex);
      futex_wait ((unsigned int *) futex, 2, private); /* Wait if *futex == 2.  */
    }
}
libc_hidden_def (__lll_lock_wait)

void
__lll_lock_wake_private (int *futex)
{
  lll_futex_wake (futex, 1, LLL_PRIVATE);
}
libc_hidden_def (__lll_lock_wake_private)

void
__lll_lock_wake (int *futex, int private)
{
  lll_futex_wake (futex, 1, private);
}
libc_hidden_def (__lll_lock_wake)

#if ENABLE_ELISION_SUPPORT
int __pthread_force_elision;
libc_hidden_data_def (__pthread_force_elision)
#endif
