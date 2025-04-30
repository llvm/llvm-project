/* Get priority ceiling setting from pthread_mutexattr_t.
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

#include <pthreadP.h>
#include <atomic.h>
#include <shlib-compat.h>

int
__pthread_mutexattr_getprioceiling (const pthread_mutexattr_t *attr,
				    int *prioceiling)
{
  const struct pthread_mutexattr *iattr;
  int ceiling;

  iattr = (const struct pthread_mutexattr *) attr;

  ceiling = ((iattr->mutexkind & PTHREAD_MUTEXATTR_PRIO_CEILING_MASK)
	     >> PTHREAD_MUTEXATTR_PRIO_CEILING_SHIFT);

  if (! ceiling)
    {
      /* See __init_sched_fifo_prio.  */
      if (atomic_load_relaxed (&__sched_fifo_min_prio) == -1)
	__init_sched_fifo_prio ();
      if (ceiling < atomic_load_relaxed (&__sched_fifo_min_prio))
	ceiling = atomic_load_relaxed (&__sched_fifo_min_prio);
    }

  *prioceiling = ceiling;

  return 0;
}
versioned_symbol (libc, __pthread_mutexattr_getprioceiling,
		  pthread_mutexattr_getprioceiling, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_4, GLIBC_2_34)
compat_symbol (libpthread, __pthread_mutexattr_getprioceiling,
               pthread_mutexattr_getprioceiling, GLIBC_2_4);
#endif
