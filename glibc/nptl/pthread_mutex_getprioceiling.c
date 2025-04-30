/* Get current priority ceiling of pthread_mutex_t.
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

#include <errno.h>
#include <pthreadP.h>
#include <shlib-compat.h>

int
__pthread_mutex_getprioceiling (const pthread_mutex_t *mutex, int *prioceiling)
{
  /* See concurrency notes regarding __kind in struct __pthread_mutex_s
     in sysdeps/nptl/bits/thread-shared-types.h.  */
  if (__builtin_expect ((atomic_load_relaxed (&(mutex->__data.__kind))
			 & PTHREAD_MUTEX_PRIO_PROTECT_NP) == 0, 0))
    return EINVAL;

  *prioceiling = (mutex->__data.__lock & PTHREAD_MUTEX_PRIO_CEILING_MASK)
		 >> PTHREAD_MUTEX_PRIO_CEILING_SHIFT;

  return 0;
}
versioned_symbol (libc, __pthread_mutex_getprioceiling,
		  pthread_mutex_getprioceiling, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_4, GLIBC_2_34)
compat_symbol (libpthread, __pthread_mutex_getprioceiling,
               pthread_mutex_getprioceiling, GLIBC_2_4);
#endif
