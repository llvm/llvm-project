/* Allocate a new thread structure.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <pt-internal.h>

/* This braindamage is necessary because the standard says that some
   of the threads functions "shall fail" if "No thread could be found
   corresponding to that specified by the given thread ID."  */

/* Thread ID lookup table.  */
struct __pthread **__pthread_threads;

/* The size of the thread ID lookup table.  */
int __pthread_max_threads;

/* The total number of thread IDs currently in use, or on the list of
   available thread IDs.  */
int __pthread_num_threads;

/* A lock for the table, and the other variables above.  */
pthread_rwlock_t __pthread_threads_lock;

/* List of thread structures corresponding to free thread IDs.  */
struct __pthread *__pthread_free_threads;
pthread_mutex_t __pthread_free_threads_lock;

static inline error_t
initialize_pthread (struct __pthread *new)
{
  error_t err;

  err = __pthread_init_specific (new);
  if (err)
    return err;

  new->nr_refs = 1;
  new->cancel_lock = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
  new->cancel_hook = NULL;
  new->cancel_hook_arg = NULL;
  new->cancel_state = PTHREAD_CANCEL_ENABLE;
  new->cancel_type = PTHREAD_CANCEL_DEFERRED;
  new->cancel_pending = 0;

  new->state_lock = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
  new->state_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;

  memset (&new->res_state, '\0', sizeof (new->res_state));

  new->tcb = NULL;

  new->next = 0;
  new->prevp = 0;

  return 0;
}


/* Allocate a new thread structure and its pthread thread ID (but not
   a kernel thread).  */
int
__pthread_alloc (struct __pthread **pthread)
{
  error_t err;

  struct __pthread *new;
  struct __pthread **threads;
  struct __pthread **old_threads;
  int max_threads;
  int new_max_threads;

  __pthread_mutex_lock (&__pthread_free_threads_lock);
  for (new = __pthread_free_threads; new; new = new->next)
    {
      /* There is no need to take NEW->STATE_LOCK: if NEW is on this
         list, then it is protected by __PTHREAD_FREE_THREADS_LOCK
         except in __pthread_dealloc where after it is added to the
         list (with the lock held), it drops the lock and then sets
         NEW->STATE and immediately stops using NEW.  */
      if (new->state == PTHREAD_TERMINATED)
	{
	  __pthread_dequeue (new);
	  break;
	}
    }
  __pthread_mutex_unlock (&__pthread_free_threads_lock);

  if (new)
    {
      if (new->tcb)
	{
	  /* Drop old values */
	  _dl_deallocate_tls (new->tcb, 1);
	}

      err = initialize_pthread (new);
      if (!err)
	*pthread = new;
      return err;
    }

  /* Allocate a new thread structure.  */
  new = malloc (sizeof (struct __pthread));
  if (new == NULL)
    return ENOMEM;

  err = initialize_pthread (new);
  if (err)
    {
      free (new);
      return err;
    }

retry:
  __pthread_rwlock_wrlock (&__pthread_threads_lock);

  if (__pthread_num_threads < __pthread_max_threads)
    {
      /* We have a free slot.  Use the slot number plus one as the
         thread ID for the new thread.  */
      new->thread = 1 + __pthread_num_threads++;
      __pthread_threads[new->thread - 1] = NULL;

      __pthread_rwlock_unlock (&__pthread_threads_lock);

      *pthread = new;
      return 0;
    }
#ifdef PTHREAD_THREADS_MAX
  else if (__pthread_num_threads >= PTHREAD_THREADS_MAX)
    {
      /* We have reached the limit on the number of threads per process.  */
      __pthread_rwlock_unlock (&__pthread_threads_lock);

      free (new);
      return EAGAIN;
    }
#endif

  /* We are going to enlarge the threads table.  Save its current
     size.  We're going to release the lock before doing the necessary
     memory allocation, since that's a potentially blocking operation.  */
  max_threads = __pthread_max_threads;

  __pthread_rwlock_unlock (&__pthread_threads_lock);

  /* Allocate a new lookup table that's twice as large.  */
  new_max_threads
      = max_threads > 0 ? max_threads * 2 : _POSIX_THREAD_THREADS_MAX;
  threads = malloc (new_max_threads * sizeof (struct __pthread *));
  if (threads == NULL)
    {
      free (new);
      return ENOMEM;
    }

  __pthread_rwlock_wrlock (&__pthread_threads_lock);

  /* Check if nobody else has already enlarged the table.  */
  if (max_threads != __pthread_max_threads)
    {
      /* Yep, they did.  */
      __pthread_rwlock_unlock (&__pthread_threads_lock);

      /* Free the newly allocated table and try again to allocate a slot.  */
      free (threads);
      goto retry;
    }

  /* Copy over the contents of the old table.  */
  memcpy (threads, __pthread_threads,
	  __pthread_max_threads * sizeof (struct __pthread *));

  /* Save the location of the old table.  We want to deallocate its
     storage after we released the lock.  */
  old_threads = __pthread_threads;

  /* Replace the table with the new one.  */
  __pthread_max_threads = new_max_threads;
  __pthread_threads = threads;

  /* And allocate ourselves one of the newly created slots.  */
  new->thread = 1 + __pthread_num_threads++;
  __pthread_threads[new->thread - 1] = NULL;

  __pthread_rwlock_unlock (&__pthread_threads_lock);

  free (old_threads);

  *pthread = new;
  return 0;
}

void
attribute_hidden
__pthread_init_static_tls (struct link_map *map)
{
  int i;

  __pthread_rwlock_wrlock (&__pthread_threads_lock);
  for (i = 0; i < __pthread_num_threads; ++i)
    {
      struct __pthread *t = __pthread_threads[i];

      if (t == NULL)
	continue;

# if TLS_TCB_AT_TP
      void *dest = (char *) t->tcb - map->l_tls_offset;
# elif TLS_DTV_AT_TP
      void *dest = (char *) t->tcb + map->l_tls_offset + TLS_PRE_TCB_SIZE;
# else
#  error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
# endif

      /* Initialize the memory.  */
      memset (__mempcpy (dest, map->l_tls_initimage, map->l_tls_initimage_size),
	      '\0', map->l_tls_blocksize - map->l_tls_initimage_size);
    }
  __pthread_rwlock_unlock (&__pthread_threads_lock);
}
