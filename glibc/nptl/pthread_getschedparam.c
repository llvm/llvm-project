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
#include <string.h>
#include "pthreadP.h"
#include <lowlevellock.h>


int
__pthread_getschedparam (pthread_t threadid, int *policy,
			 struct sched_param *param)
{
  struct pthread *pd = (struct pthread *) threadid;

  /* Make sure the descriptor is valid.  */
  if (INVALID_TD_P (pd))
    /* Not a valid thread handle.  */
    return ESRCH;

  int result = 0;

  /* See CREATE THREAD NOTES in nptl/pthread_create.c.  */
  lll_lock (pd->lock, LLL_PRIVATE);

  /* The library is responsible for maintaining the values at all
     times.  If the user uses an interface other than
     pthread_setschedparam to modify the scheduler setting it is not
     the library's problem.  In case the descriptor's values have
     not yet been retrieved do it now.  */
  if ((pd->flags & ATTR_FLAG_SCHED_SET) == 0)
    {
      if (__sched_getparam (pd->tid, &pd->schedparam) != 0)
	result = 1;
      else
	pd->flags |= ATTR_FLAG_SCHED_SET;
    }

  if ((pd->flags & ATTR_FLAG_POLICY_SET) == 0)
    {
      pd->schedpolicy = __sched_getscheduler (pd->tid);
      if (pd->schedpolicy == -1)
	result = 1;
      else
	pd->flags |= ATTR_FLAG_POLICY_SET;
    }

  if (result == 0)
    {
      *policy = pd->schedpolicy;
      memcpy (param, &pd->schedparam, sizeof (struct sched_param));
    }

  lll_unlock (pd->lock, LLL_PRIVATE);

  return result;
}
libc_hidden_def (__pthread_getschedparam)
strong_alias (__pthread_getschedparam, pthread_getschedparam)
