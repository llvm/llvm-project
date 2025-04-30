/* System specific fork hooks.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _FORK_H
#define _FORK_H

#include <assert.h>
#include <kernel-posix-timers.h>
#include <ldsodefs.h>
#include <list.h>
#include <mqueue.h>
#include <pthreadP.h>
#include <sysdep.h>

static inline void
fork_system_setup (void)
{
  /* See __pthread_once.  */
  __fork_generation += __PTHREAD_ONCE_FORK_GEN_INCR;
}

static void
fork_system_setup_after_fork (void)
{
  /* There is one thread running.  */
  __nptl_nthreads = 1;

  /* Initialize thread library locks.  */
  GL (dl_stack_cache_lock) = LLL_LOCK_INITIALIZER;
  __default_pthread_attr_lock = LLL_LOCK_INITIALIZER;

  call_function_static_weak (__mq_notify_fork_subprocess);
  call_function_static_weak (__timer_fork_subprocess);
}

/* In case of a fork() call the memory allocation in the child will be
   the same but only one thread is running.  All stacks except that of
   the one running thread are not used anymore.  We have to recycle
   them.  */
static void
reclaim_stacks (void)
{
  struct pthread *self = (struct pthread *) THREAD_SELF;

  /* No locking necessary.  The caller is the only stack in use.  But
     we have to be aware that we might have interrupted a list
     operation.  */

  if (GL (dl_in_flight_stack) != 0)
    {
      bool add_p = GL (dl_in_flight_stack) & 1;
      list_t *elem = (list_t *) (GL (dl_in_flight_stack) & ~(uintptr_t) 1);

      if (add_p)
	{
	  /* We always add at the beginning of the list.  So in this case we
	     only need to check the beginning of these lists to see if the
	     pointers at the head of the list are inconsistent.  */
	  list_t *l = NULL;

	  if (GL (dl_stack_used).next->prev != &GL (dl_stack_used))
	    l = &GL (dl_stack_used);
	  else if (GL (dl_stack_cache).next->prev != &GL (dl_stack_cache))
	    l = &GL (dl_stack_cache);

	  if (l != NULL)
	    {
	      assert (l->next->prev == elem);
	      elem->next = l->next;
	      elem->prev = l;
	      l->next = elem;
	    }
	}
      else
	{
	  /* We can simply always replay the delete operation.  */
	  elem->next->prev = elem->prev;
	  elem->prev->next = elem->next;
	}

      GL (dl_in_flight_stack) = 0;
    }

  /* Mark all stacks except the still running one as free.  */
  list_t *runp;
  list_for_each (runp, &GL (dl_stack_used))
    {
      struct pthread *curp = list_entry (runp, struct pthread, list);
      if (curp != self)
	{
	  /* This marks the stack as free.  */
	  curp->tid = 0;

	  /* Account for the size of the stack.  */
	  GL (dl_stack_cache_actsize) += curp->stackblock_size;

	  if (curp->specific_used)
	    {
	      /* Clear the thread-specific data.  */
	      memset (curp->specific_1stblock, '\0',
		      sizeof (curp->specific_1stblock));

	      curp->specific_used = false;

	      for (size_t cnt = 1; cnt < PTHREAD_KEY_1STLEVEL_SIZE; ++cnt)
		if (curp->specific[cnt] != NULL)
		  {
		    memset (curp->specific[cnt], '\0',
			    sizeof (curp->specific_1stblock));

		    /* We have allocated the block which we do not
		       free here so re-set the bit.  */
		    curp->specific_used = true;
		  }
	    }
	}
    }

  /* Add the stack of all running threads to the cache.  */
  list_splice (&GL (dl_stack_used), &GL (dl_stack_cache));

  /* Remove the entry for the current thread to from the cache list
     and add it to the list of running threads.  Which of the two
     lists is decided by the user_stack flag.  */
  list_del (&self->list);

  /* Re-initialize the lists for all the threads.  */
  INIT_LIST_HEAD (&GL (dl_stack_used));
  INIT_LIST_HEAD (&GL (dl_stack_user));

  if (__glibc_unlikely (THREAD_GETMEM (self, user_stack)))
    list_add (&self->list, &GL (dl_stack_user));
  else
    list_add (&self->list, &GL (dl_stack_used));
}


#endif
