/* pthread_mutex_consistent.  Hurd version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <assert.h>
#include <pt-internal.h>
#include "pt-mutex.h"
#include <hurdlock.h>

int
pthread_mutex_consistent (pthread_mutex_t *mtxp)
{
  int ret = EINVAL;
  unsigned int val = mtxp->__lock;

  if ((mtxp->__flags & PTHREAD_MUTEX_ROBUST) != 0
      && (val & LLL_DEAD_OWNER) != 0
      && atomic_compare_and_exchange_bool_acq (&mtxp->__lock,
					       __getpid () | LLL_WAITERS,
					       val) == 0)
    {
      /* The mutex is now ours, and it's consistent. */
      mtxp->__owner_id = _pthread_self ()->thread;
      mtxp->__cnt = 1;
      ret = 0;
    }

  return ret;
}

weak_alias (pthread_mutex_consistent, pthread_mutex_consistent_np)
