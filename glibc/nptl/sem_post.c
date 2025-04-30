/* sem_post -- post to a POSIX semaphore.  Generic futex-using version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <atomic.h>
#include <errno.h>
#include <sysdep.h>
#include <lowlevellock.h>	/* lll_futex* used by the old code.  */
#include <futex-internal.h>
#include <internaltypes.h>
#include <semaphore.h>

#include <shlib-compat.h>


/* See sem_wait for an explanation of the algorithm.  */
int
__new_sem_post (sem_t *sem)
{
  struct new_sem *isem = (struct new_sem *) sem;
  int private = isem->private;

#if __HAVE_64B_ATOMICS
  /* Add a token to the semaphore.  We use release MO to make sure that a
     thread acquiring this token synchronizes with us and other threads that
     added tokens before (the release sequence includes atomic RMW operations
     by other threads).  */
  /* TODO Use atomic_fetch_add to make it scale better than a CAS loop?  */
  uint64_t d = atomic_load_relaxed (&isem->data);
  do
    {
      if ((d & SEM_VALUE_MASK) == SEM_VALUE_MAX)
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}
    }
  while (!atomic_compare_exchange_weak_release (&isem->data, &d, d + 1));

  /* If there is any potentially blocked waiter, wake one of them.  */
  if ((d >> SEM_NWAITERS_SHIFT) > 0)
    futex_wake (((unsigned int *) &isem->data) + SEM_VALUE_OFFSET, 1, private);
#else
  /* Add a token to the semaphore.  Similar to 64b version.  */
  unsigned int v = atomic_load_relaxed (&isem->value);
  do
    {
      if ((v >> SEM_VALUE_SHIFT) == SEM_VALUE_MAX)
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}
    }
  while (!atomic_compare_exchange_weak_release
	 (&isem->value, &v, v + (1 << SEM_VALUE_SHIFT)));

  /* If there is any potentially blocked waiter, wake one of them.  */
  if ((v & SEM_NWAITERS_MASK) != 0)
    futex_wake (&isem->value, 1, private);
#endif

  return 0;
}
versioned_symbol (libpthread, __new_sem_post, sem_post, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __new_sem_post, sem_post, GLIBC_2_1);
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_1)
int
attribute_compat_text_section
__old_sem_post (sem_t *sem)
{
  unsigned int *futex = (unsigned int *) sem;

  /* We must need to synchronize with consumers of this token, so the atomic
     increment must have release MO semantics.  */
  atomic_write_barrier ();
  (void) atomic_increment_val (futex);
  /* We always have to assume it is a shared semaphore.  */
  futex_wake (futex, 1, LLL_SHARED);
  return 0;
}
compat_symbol (libpthread, __old_sem_post, sem_post, GLIBC_2_0);
#endif
