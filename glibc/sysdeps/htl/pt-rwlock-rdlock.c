/* Acquire a rwlock for reading.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <pt-internal.h>

/* Implemented in pt-rwlock-timedrdlock.c.  */
extern int __pthread_rwlock_timedrdlock_internal (struct __pthread_rwlock
						  *rwlock,
						  clockid_t clockid,
						  const struct timespec
						  *abstime);

/* Acquire RWLOCK for reading, block if we can't get it.  */
int
__pthread_rwlock_rdlock (struct __pthread_rwlock *rwlock)
{
  return __pthread_rwlock_timedrdlock_internal (rwlock, -1, 0);
}
weak_alias (__pthread_rwlock_rdlock, pthread_rwlock_rdlock);
