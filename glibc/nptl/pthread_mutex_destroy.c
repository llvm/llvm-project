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

#include <errno.h>
#include "pthreadP.h"
#include <shlib-compat.h>
#include <stap-probe.h>


int
___pthread_mutex_destroy (pthread_mutex_t *mutex)
{
  LIBC_PROBE (mutex_destroy, 1, mutex);

  /* See concurrency notes regarding __kind in struct __pthread_mutex_s
     in sysdeps/nptl/bits/thread-shared-types.h.  */
  if ((atomic_load_relaxed (&(mutex->__data.__kind))
       & PTHREAD_MUTEX_ROBUST_NORMAL_NP) == 0
      && mutex->__data.__nusers != 0)
    return EBUSY;

  /* Set to an invalid value.  Relaxed MO is enough as it is undefined behavior
     if the mutex is used after it has been destroyed.  But you can reinitialize
     it with pthread_mutex_init.  */
  atomic_store_relaxed (&(mutex->__data.__kind), -1);

  return 0;
}
versioned_symbol (libc, ___pthread_mutex_destroy, pthread_mutex_destroy,
                  GLIBC_2_0);
libc_hidden_ver (___pthread_mutex_destroy, __pthread_mutex_destroy)
#ifndef SHARED
strong_alias (___pthread_mutex_destroy, __pthread_mutex_destroy)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutex_destroy, __pthread_mutex_destroy,
               GLIBC_2_0);
#endif
