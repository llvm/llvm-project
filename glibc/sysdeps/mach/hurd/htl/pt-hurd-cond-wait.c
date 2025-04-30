/* pthread_hurd_cond_wait.  Hurd-specific wait on a condition.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <assert.h>
#include <hurd/signal.h>

#include <pt-internal.h>

/* Implemented in pt-hurd-cond-timedwait.c.  */
extern int __pthread_hurd_cond_timedwait_internal (pthread_cond_t *cond,
						   pthread_mutex_t *mutex,
						   const struct timespec
						   *abstime);

int
__pthread_hurd_cond_wait_np (pthread_cond_t *cond, pthread_mutex_t *mutex)
{
  error_t err;

  err = __pthread_hurd_cond_timedwait_internal (cond, mutex, NULL);
  return err == EINTR;
}

strong_alias (__pthread_hurd_cond_wait_np, pthread_hurd_cond_wait_np);
