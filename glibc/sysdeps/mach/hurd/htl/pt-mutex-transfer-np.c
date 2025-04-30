/* pthread_mutex_transfer_np.  Transfer mutex ownership to another thread.
   Hurd version.
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
__pthread_mutex_transfer_np (pthread_mutex_t *mtxp, pthread_t th)
{
  struct __pthread *self = _pthread_self ();
  struct __pthread *pt = __pthread_getid (th);

  if (pt == NULL)
    return ESRCH;
  else if (pt == self)
    return 0;

  int ret = 0;
  int flags = mtxp->__flags & GSYNC_SHARED;

  switch (MTX_TYPE (mtxp))
    {
    case PT_MTX_NORMAL:
      break;

    case PT_MTX_RECURSIVE:
    case PT_MTX_ERRORCHECK:
      if (!mtx_owned_p (mtxp, self, flags))
	ret = EPERM;
      else
	mtx_set_owner (mtxp, pt, flags);

      break;

    case PT_MTX_NORMAL | PTHREAD_MUTEX_ROBUST:
    case PT_MTX_RECURSIVE | PTHREAD_MUTEX_ROBUST:
    case PT_MTX_ERRORCHECK | PTHREAD_MUTEX_ROBUST:
      /* Note that this can be used to transfer an inconsistent
       * mutex as well. The new owner will still have the same
       * flags as the original. */
      if (mtxp->__owner_id != self->thread
	  || (int) (mtxp->__lock & LLL_OWNER_MASK) != __getpid ())
	ret = EPERM;
      else
	mtxp->__owner_id = pt->thread;

      break;

    default:
      ret = EINVAL;
    }

  return ret;
}

weak_alias (__pthread_mutex_transfer_np, pthread_mutex_transfer_np)
