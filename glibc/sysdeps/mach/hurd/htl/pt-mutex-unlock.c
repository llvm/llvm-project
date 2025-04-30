/* pthread_mutex_unlock.  Hurd version.
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
__pthread_mutex_unlock (pthread_mutex_t *mtxp)
{
  struct __pthread *self;
  int ret = 0, flags = mtxp->__flags & GSYNC_SHARED;

  switch (MTX_TYPE (mtxp))
    {
    case PT_MTX_NORMAL:
      lll_unlock (mtxp->__lock, flags);
      break;

    case PT_MTX_RECURSIVE:
      self = _pthread_self ();
      if (!mtx_owned_p (mtxp, self, flags))
	ret = EPERM;
      else if (--mtxp->__cnt == 0)
	{
	  mtxp->__owner_id = mtxp->__shpid = 0;
	  lll_unlock (mtxp->__lock, flags);
	}

      break;

    case PT_MTX_ERRORCHECK:
      self = _pthread_self ();
      if (!mtx_owned_p (mtxp, self, flags))
	ret = EPERM;
      else
	{
	  mtxp->__owner_id = mtxp->__shpid = 0;
	  lll_unlock (mtxp->__lock, flags);
	}

      break;

    case PT_MTX_NORMAL | PTHREAD_MUTEX_ROBUST:
    case PT_MTX_RECURSIVE | PTHREAD_MUTEX_ROBUST:
    case PT_MTX_ERRORCHECK | PTHREAD_MUTEX_ROBUST:
      self = _pthread_self ();
      if (mtxp->__owner_id == NOTRECOVERABLE_ID)
	;			/* Nothing to do. */
      else if (mtxp->__owner_id != self->thread
	       || (int) (mtxp->__lock & LLL_OWNER_MASK) != __getpid ())
	ret = EPERM;
      else if (--mtxp->__cnt == 0)
	{
	  /* Release the lock. If it's in an inconsistent
	   * state, mark it as irrecoverable. */
	  mtxp->__owner_id = ((mtxp->__lock & LLL_DEAD_OWNER)
			      ? NOTRECOVERABLE_ID : 0);
	  lll_robust_unlock (mtxp->__lock, flags);
	}

      break;

    default:
      ret = EINVAL;
      break;
    }

  return ret;
}

hidden_def (__pthread_mutex_unlock)
strong_alias (__pthread_mutex_unlock, _pthread_mutex_unlock)
weak_alias (__pthread_mutex_unlock, pthread_mutex_unlock)
