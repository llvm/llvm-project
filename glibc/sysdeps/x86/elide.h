/* elide.h: Generic lock elision support.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
#ifndef ELIDE_H
#define ELIDE_H 1

#include <hle.h>
#include <elision-conf.h>
#include <atomic.h>


/* Adapt elision with ADAPT_COUNT and STATUS and decide retries.  */

static inline bool
elision_adapt(signed char *adapt_count, unsigned int status)
{
  if (status & _XABORT_RETRY)
    return false;
  if ((status & _XABORT_EXPLICIT)
      && _XABORT_CODE (status) == _ABORT_LOCK_BUSY)
    {
      /* Right now we skip here.  Better would be to wait a bit
	 and retry.  This likely needs some spinning. Be careful
	 to avoid writing the lock.
	 Using relaxed MO and separate atomic accesses is sufficient because
	 adapt_count is just a hint.  */
      if (atomic_load_relaxed (adapt_count) != __elision_aconf.skip_lock_busy)
	atomic_store_relaxed (adapt_count, __elision_aconf.skip_lock_busy);
    }
  /* Internal abort.  There is no chance for retry.
     Use the normal locking and next time use lock.
     Be careful to avoid writing to the lock.  See above for MO.  */
  else if (atomic_load_relaxed (adapt_count)
      != __elision_aconf.skip_lock_internal_abort)
    atomic_store_relaxed (adapt_count,
	__elision_aconf.skip_lock_internal_abort);
  return true;
}

/* is_lock_free must be executed inside the transaction */

/* Returns true if lock defined by IS_LOCK_FREE was elided.
   ADAPT_COUNT is a per-lock state variable; it must be accessed atomically
   to avoid data races but is just a hint, so using relaxed MO and separate
   atomic loads and stores instead of atomic read-modify-write operations is
   sufficient.  */

#define ELIDE_LOCK(adapt_count, is_lock_free)			\
  ({								\
    int ret = 0;						\
								\
    if (atomic_load_relaxed (&(adapt_count)) <= 0)		\
      {								\
        for (int i = __elision_aconf.retry_try_xbegin; i > 0; i--) \
          {							\
            unsigned int status;				\
	    if ((status = _xbegin ()) == _XBEGIN_STARTED)	\
	      {							\
	        if (is_lock_free)				\
	          {						\
		    ret = 1;					\
		    break;					\
	          }						\
	        _xabort (_ABORT_LOCK_BUSY);			\
	      }							\
	    if (!elision_adapt (&(adapt_count), status))	\
	      break;						\
          }							\
      }								\
    else 							\
      atomic_store_relaxed (&(adapt_count),			\
	  atomic_load_relaxed (&(adapt_count)) - 1);		\
    ret;							\
  })

/* Returns true if lock defined by IS_LOCK_FREE was try-elided.
   ADAPT_COUNT is a per-lock state variable.  */

#define ELIDE_TRYLOCK(adapt_count, is_lock_free, write) ({	\
  int ret = 0;						\
  if (__elision_aconf.retry_try_xbegin > 0)		\
    {  							\
      if (write)					\
        _xabort (_ABORT_NESTED_TRYLOCK);		\
      ret = ELIDE_LOCK (adapt_count, is_lock_free);     \
    }							\
    ret;						\
    })

/* Returns true if lock defined by IS_LOCK_FREE was elided.  The call
   to _xend crashes if the application incorrectly tries to unlock a
   lock which has not been locked.  */

#define ELIDE_UNLOCK(is_lock_free)		\
  ({						\
  int ret = 0;					\
  if (is_lock_free)				\
    {						\
      _xend ();					\
      ret = 1;					\
    }						\
  ret;						\
  })

#endif
