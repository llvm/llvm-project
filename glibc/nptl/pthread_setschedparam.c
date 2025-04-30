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
#include <sched.h>
#include <string.h>
#include "pthreadP.h"
#include <lowlevellock.h>


int
__pthread_setschedparam (pthread_t threadid, int policy,
			 const struct sched_param *param)
{
  struct pthread *pd = (struct pthread *) threadid;

  /* Make sure the descriptor is valid.  */
  if (INVALID_TD_P (pd))
    /* Not a valid thread handle.  */
    return ESRCH;

  int result = 0;

  /* See CREATE THREAD NOTES in nptl/pthread_create.c.  */
  lll_lock (pd->lock, LLL_PRIVATE);

  struct sched_param p;
  const struct sched_param *orig_param = param;

  /* If the thread should have higher priority because of some
     PTHREAD_PRIO_PROTECT mutexes it holds, adjust the priority.  */
  if (__builtin_expect (pd->tpp != NULL, 0)
      && pd->tpp->priomax > param->sched_priority)
    {
      p = *param;
      p.sched_priority = pd->tpp->priomax;
      param = &p;
    }

  /* Try to set the scheduler information.  */
  if (__builtin_expect (__sched_setscheduler (pd->tid, policy,
					      param) == -1, 0))
    result = errno;
  else
    {
      /* We succeeded changing the kernel information.  Reflect this
	 change in the thread descriptor.  */
      pd->schedpolicy = policy;
      memcpy (&pd->schedparam, orig_param, sizeof (struct sched_param));
      pd->flags |= ATTR_FLAG_SCHED_SET | ATTR_FLAG_POLICY_SET;
    }

  lll_unlock (pd->lock, LLL_PRIVATE);

  return result;
}
strong_alias (__pthread_setschedparam, pthread_setschedparam)
