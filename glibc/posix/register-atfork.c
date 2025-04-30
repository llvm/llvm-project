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

#include <libc-lock.h>
#include <stdbool.h>
#include <register-atfork.h>

#define DYNARRAY_ELEMENT           struct fork_handler
#define DYNARRAY_STRUCT            fork_handler_list
#define DYNARRAY_PREFIX            fork_handler_list_
#define DYNARRAY_INITIAL_SIZE      48
#include <malloc/dynarray-skeleton.c>

static struct fork_handler_list fork_handlers;
static bool fork_handler_init = false;

static int atfork_lock = LLL_LOCK_INITIALIZER;

int
__register_atfork (void (*prepare) (void), void (*parent) (void),
		   void (*child) (void), void *dso_handle)
{
  lll_lock (atfork_lock, LLL_PRIVATE);

  if (!fork_handler_init)
    {
      fork_handler_list_init (&fork_handlers);
      fork_handler_init = true;
    }

  struct fork_handler *newp = fork_handler_list_emplace (&fork_handlers);
  if (newp != NULL)
    {
      newp->prepare_handler = prepare;
      newp->parent_handler = parent;
      newp->child_handler = child;
      newp->dso_handle = dso_handle;
    }

  /* Release the lock.  */
  lll_unlock (atfork_lock, LLL_PRIVATE);

  return newp == NULL ? ENOMEM : 0;
}
libc_hidden_def (__register_atfork)

static struct fork_handler *
fork_handler_list_find (struct fork_handler_list *fork_handlers,
			void *dso_handle)
{
  for (size_t i = 0; i < fork_handler_list_size (fork_handlers); i++)
    {
      struct fork_handler *elem = fork_handler_list_at (fork_handlers, i);
      if (elem->dso_handle == dso_handle)
	return elem;
    }
  return NULL;
}

void
__unregister_atfork (void *dso_handle)
{
  lll_lock (atfork_lock, LLL_PRIVATE);

  struct fork_handler *first = fork_handler_list_find (&fork_handlers,
						       dso_handle);
  /* Removing is done by shifting the elements in the way the elements
     that are not to be removed appear in the beginning in dynarray.
     This avoid the quadradic run-time if a naive strategy to remove and
     shift one element at time.  */
  if (first != NULL)
    {
      struct fork_handler *new_end = first;
      first++;
      for (; first != fork_handler_list_end (&fork_handlers); ++first)
	{
	  if (first->dso_handle != dso_handle)
	    {
	      *new_end = *first;
	      ++new_end;
	    }
	}

      ptrdiff_t removed = first - new_end;
      for (size_t i = 0; i < removed; i++)
	fork_handler_list_remove_last (&fork_handlers);
    }

  lll_unlock (atfork_lock, LLL_PRIVATE);
}

void
__run_fork_handlers (enum __run_fork_handler_type who, _Bool do_locking)
{
  struct fork_handler *runp;

  if (who == atfork_run_prepare)
    {
      if (do_locking)
	lll_lock (atfork_lock, LLL_PRIVATE);
      size_t sl = fork_handler_list_size (&fork_handlers);
      for (size_t i = sl; i > 0; i--)
	{
	  runp = fork_handler_list_at (&fork_handlers, i - 1);
	  if (runp->prepare_handler != NULL)
	    runp->prepare_handler ();
	}
    }
  else
    {
      size_t sl = fork_handler_list_size (&fork_handlers);
      for (size_t i = 0; i < sl; i++)
	{
	  runp = fork_handler_list_at (&fork_handlers, i);
	  if (who == atfork_run_child && runp->child_handler)
	    runp->child_handler ();
	  else if (who == atfork_run_parent && runp->parent_handler)
	    runp->parent_handler ();
	}
      if (do_locking)
	lll_unlock (atfork_lock, LLL_PRIVATE);
    }
}


libc_freeres_fn (free_mem)
{
  lll_lock (atfork_lock, LLL_PRIVATE);

  fork_handler_list_free (&fork_handlers);

  lll_unlock (atfork_lock, LLL_PRIVATE);
}
