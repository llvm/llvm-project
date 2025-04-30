/* Thread Priority Protect helpers.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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

#include <assert.h>
#include <atomic.h>
#include <errno.h>
#include <pthreadP.h>
#include <sched.h>
#include <stdlib.h>
#include <atomic.h>

int __sched_fifo_min_prio = -1;
libc_hidden_data_def (__sched_fifo_min_prio)
int __sched_fifo_max_prio = -1;
libc_hidden_data_def (__sched_fifo_max_prio)

/* We only want to initialize __sched_fifo_min_prio and __sched_fifo_max_prio
   once.  The standard solution would be similar to pthread_once, but then
   readers would need to use an acquire fence.  In this specific case,
   initialization is comprised of just idempotent writes to two variables
   that have an initial value of -1.  Therefore, we can treat each variable as
   a separate, at-least-once initialized value.  This enables using just
   relaxed MO loads and stores, but requires that consumers check for
   initialization of each value that is to be used; see
   __pthread_tpp_change_priority for an example.
 */
void
__init_sched_fifo_prio (void)
{
  atomic_store_relaxed (&__sched_fifo_max_prio,
			__sched_get_priority_max (SCHED_FIFO));
  atomic_store_relaxed (&__sched_fifo_min_prio,
			__sched_get_priority_min (SCHED_FIFO));
}
libc_hidden_def (__init_sched_fifo_prio)

int
__pthread_tpp_change_priority (int previous_prio, int new_prio)
{
  struct pthread *self = THREAD_SELF;
  struct priority_protection_data *tpp = THREAD_GETMEM (self, tpp);
  int fifo_min_prio = atomic_load_relaxed (&__sched_fifo_min_prio);
  int fifo_max_prio = atomic_load_relaxed (&__sched_fifo_max_prio);

  if (tpp == NULL)
    {
      /* See __init_sched_fifo_prio.  We need both the min and max prio,
         so need to check both, and run initialization if either one is
         not initialized.  The memory model's write-read coherence rule
         makes this work.  */
      if (fifo_min_prio == -1 || fifo_max_prio == -1)
	{
	  __init_sched_fifo_prio ();
	  fifo_min_prio = atomic_load_relaxed (&__sched_fifo_min_prio);
	  fifo_max_prio = atomic_load_relaxed (&__sched_fifo_max_prio);
	}

      size_t size = sizeof *tpp;
      size += (fifo_max_prio - fifo_min_prio + 1)
	      * sizeof (tpp->priomap[0]);
      tpp = calloc (size, 1);
      if (tpp == NULL)
	return ENOMEM;
      tpp->priomax = fifo_min_prio - 1;
      THREAD_SETMEM (self, tpp, tpp);
    }

  assert (new_prio == -1
	  || (new_prio >= fifo_min_prio
	      && new_prio <= fifo_max_prio));
  assert (previous_prio == -1
	  || (previous_prio >= fifo_min_prio
	      && previous_prio <= fifo_max_prio));

  int priomax = tpp->priomax;
  int newpriomax = priomax;
  if (new_prio != -1)
    {
      if (tpp->priomap[new_prio - fifo_min_prio] + 1 == 0)
	return EAGAIN;
      ++tpp->priomap[new_prio - fifo_min_prio];
      if (new_prio > priomax)
	newpriomax = new_prio;
    }

  if (previous_prio != -1)
    {
      if (--tpp->priomap[previous_prio - fifo_min_prio] == 0
	  && priomax == previous_prio
	  && previous_prio > new_prio)
	{
	  int i;
	  for (i = previous_prio - 1; i >= fifo_min_prio; --i)
	    if (tpp->priomap[i - fifo_min_prio])
	      break;
	  newpriomax = i;
	}
    }

  if (priomax == newpriomax)
    return 0;

  /* See CREATE THREAD NOTES in nptl/pthread_create.c.  */
  lll_lock (self->lock, LLL_PRIVATE);

  tpp->priomax = newpriomax;

  int result = 0;

  if ((self->flags & ATTR_FLAG_SCHED_SET) == 0)
    {
      if (__sched_getparam (self->tid, &self->schedparam) != 0)
	result = errno;
      else
	self->flags |= ATTR_FLAG_SCHED_SET;
    }

  if ((self->flags & ATTR_FLAG_POLICY_SET) == 0)
    {
      self->schedpolicy = __sched_getscheduler (self->tid);
      if (self->schedpolicy == -1)
	result = errno;
      else
	self->flags |= ATTR_FLAG_POLICY_SET;
    }

  if (result == 0)
    {
      struct sched_param sp = self->schedparam;
      if (sp.sched_priority < newpriomax || sp.sched_priority < priomax)
	{
	  if (sp.sched_priority < newpriomax)
	    sp.sched_priority = newpriomax;

	  if (__sched_setscheduler (self->tid, self->schedpolicy, &sp) < 0)
	    result = errno;
	}
    }

  lll_unlock (self->lock, LLL_PRIVATE);

  return result;
}
libc_hidden_def (__pthread_tpp_change_priority)

int
__pthread_current_priority (void)
{
  struct pthread *self = THREAD_SELF;
  if ((self->flags & (ATTR_FLAG_POLICY_SET | ATTR_FLAG_SCHED_SET))
      == (ATTR_FLAG_POLICY_SET | ATTR_FLAG_SCHED_SET))
    return self->schedparam.sched_priority;

  int result = 0;

  /* See CREATE THREAD NOTES in nptl/pthread_create.c.  */
  lll_lock (self->lock, LLL_PRIVATE);

  if ((self->flags & ATTR_FLAG_SCHED_SET) == 0)
    {
      if (__sched_getparam (self->tid, &self->schedparam) != 0)
	result = -1;
      else
	self->flags |= ATTR_FLAG_SCHED_SET;
    }

  if ((self->flags & ATTR_FLAG_POLICY_SET) == 0)
    {
      self->schedpolicy = __sched_getscheduler (self->tid);
      if (self->schedpolicy == -1)
	result = -1;
      else
	self->flags |= ATTR_FLAG_POLICY_SET;
    }

  if (result != -1)
    result = self->schedparam.sched_priority;

  lll_unlock (self->lock, LLL_PRIVATE);

  return result;
}
libc_hidden_def (__pthread_current_priority)
