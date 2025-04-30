/* elide.h: Generic lock elision support for powerpc.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef ELIDE_PPC_H
# define ELIDE_PPC_H

# include <htm.h>
# include <elision-conf.h>

/* Get the new value of adapt_count according to the elision
   configurations.  Returns true if the system should retry again or false
   otherwise.  */
static inline bool
__get_new_count (uint8_t *adapt_count, int attempt)
{
  /* A persistent failure indicates that a retry will probably
     result in another failure.  Use normal locking now and
     for the next couple of calls.  */
  if (_TEXASRU_FAILURE_PERSISTENT (__builtin_get_texasru ()))
    {
      if (__elision_aconf.skip_lock_internal_abort > 0)
	*adapt_count = __elision_aconf.skip_lock_internal_abort;
      return false;
    }
  /* Same logic as above, but for a number of temporary failures in a
     a row.  */
  else if (attempt <= 1 && __elision_aconf.skip_lock_out_of_tbegin_retries > 0
	   && __elision_aconf.try_tbegin > 0)
    *adapt_count = __elision_aconf.skip_lock_out_of_tbegin_retries;
  return true;
}

/* CONCURRENCY NOTES:

   The evaluation of the macro expression is_lock_free encompasses one or
   more loads from memory locations that are concurrently modified by other
   threads.  For lock elision to work, this evaluation and the rest of the
   critical section protected by the lock must be atomic because an
   execution with lock elision must be equivalent to an execution in which
   the lock would have been actually acquired and released.  Therefore, we
   evaluate is_lock_free inside of the transaction that represents the
   critical section for which we want to use lock elision, which ensures
   the atomicity that we require.  */

/* Returns 0 if the lock defined by is_lock_free was elided.
   ADAPT_COUNT is a per-lock state variable.  */
# define ELIDE_LOCK(adapt_count, is_lock_free)				\
  ({									\
    int ret = 0;							\
    if (adapt_count > 0)						\
      (adapt_count)--;							\
    else								\
      for (int i = __elision_aconf.try_tbegin; i > 0; i--)		\
	{								\
	  if (__libc_tbegin (0))					\
	    {								\
	      if (is_lock_free)						\
		{							\
		  ret = 1;						\
		  break;						\
		}							\
	      __libc_tabort (_ABORT_LOCK_BUSY);				\
	    }								\
	  else								\
	    if (!__get_new_count (&adapt_count,i))			\
	      break;							\
	}								\
    ret;								\
  })

# define ELIDE_TRYLOCK(adapt_count, is_lock_free, write)	\
  ({								\
    int ret = 0;						\
    if (__elision_aconf.try_tbegin > 0)				\
      {								\
	if (write)						\
	  __libc_tabort (_ABORT_NESTED_TRYLOCK);		\
	ret = ELIDE_LOCK (adapt_count, is_lock_free);		\
      }								\
    ret;							\
  })


static inline bool
__elide_unlock (int is_lock_free)
{
  if (is_lock_free)
    {
      /* This code is expected to crash when trying to unlock a lock not
	 held by this thread.  More information is available in the
	 __pthread_rwlock_unlock() implementation.  */
      __libc_tend (0);
      return true;
    }
  return false;
}

# define ELIDE_UNLOCK(is_lock_free) \
  __elide_unlock (is_lock_free)

#endif
