/* Cancel a thread.
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

int
pthread_cancel (pthread_t t)
{
  int err = 0;
  struct __pthread *p;

  p = __pthread_getid (t);
  if (p == NULL)
    return ESRCH;

  __pthread_mutex_lock (&p->cancel_lock);
  if (p->cancel_pending)
    {
      __pthread_mutex_unlock (&p->cancel_lock);
      return 0;
    }

  p->cancel_pending = 1;

  if (p->cancel_state != PTHREAD_CANCEL_ENABLE)
    {
      __pthread_mutex_unlock (&p->cancel_lock);
      return 0;
    }

  if (p->cancel_type == PTHREAD_CANCEL_ASYNCHRONOUS)
    /* CANCEL_LOCK is unlocked by this call.  */
    err = __pthread_do_cancel (p);
  else
    {
      if (p->cancel_hook != NULL)
	/* Thread blocking on a cancellation point.  Invoke hook to unblock.
	   See __pthread_cond_timedwait_internal.  */
	p->cancel_hook (p->cancel_hook_arg);

      __pthread_mutex_unlock (&p->cancel_lock);
    }

  return err;
}
