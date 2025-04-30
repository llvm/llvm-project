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

#include "pthread_rwlock_common.c"

/* See pthread_rwlock_common.c.  */
int
___pthread_rwlock_wrlock (pthread_rwlock_t *rwlock)
{
  LIBC_PROBE (wrlock_entry, 1, rwlock);

  int result = __pthread_rwlock_wrlock_full64 (rwlock, CLOCK_REALTIME, NULL);
  LIBC_PROBE (wrlock_acquire_write, 1, rwlock);
  return result;
}
versioned_symbol (libc, ___pthread_rwlock_wrlock, pthread_rwlock_wrlock,
		  GLIBC_2_34);
strong_alias (___pthread_rwlock_wrlock, __pthread_rwlock_wrlock)
libc_hidden_ver (___pthread_rwlock_wrlock, __pthread_rwlock_wrlock)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_rwlock_wrlock, pthread_rwlock_wrlock,
	       GLIBC_2_1);
#endif
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_rwlock_wrlock, __pthread_rwlock_wrlock,
	       GLIBC_2_2);
#endif
