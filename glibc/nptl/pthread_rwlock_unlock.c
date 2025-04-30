/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky <schwidefsky@de.ibm.com>, 2003.

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
#include <futex-internal.h>
#include <pthread.h>
#include <pthreadP.h>
#include <stap-probe.h>

#include "pthread_rwlock_common.c"

/* See pthread_rwlock_common.c for an overview.  */
int
___pthread_rwlock_unlock (pthread_rwlock_t *rwlock)
{
  LIBC_PROBE (rwlock_unlock, 1, rwlock);

  /* We distinguish between having acquired a read vs. a write lock by looking
     at the writer TID.  If it's equal to our TID, we must be the writer
     because nobody else can have stored this value.  Also, if we are a
     reader, we will read from the wrunlock store with value 0 by the most
     recent writer because that writer happens-before us.  */
  if (atomic_load_relaxed (&rwlock->__data.__cur_writer)
      == THREAD_GETMEM (THREAD_SELF, tid))
      __pthread_rwlock_wrunlock (rwlock);
  else
    __pthread_rwlock_rdunlock (rwlock);
  return 0;
}
versioned_symbol (libc, ___pthread_rwlock_unlock, pthread_rwlock_unlock,
		  GLIBC_2_34);
strong_alias (___pthread_rwlock_unlock, __pthread_rwlock_unlock)
libc_hidden_ver (___pthread_rwlock_unlock, __pthread_rwlock_unlock)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_rwlock_unlock, pthread_rwlock_unlock,
	       GLIBC_2_1);
#endif
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_rwlock_unlock, __pthread_rwlock_unlock,
	       GLIBC_2_2);
#endif
