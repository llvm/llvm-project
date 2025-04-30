/* Post a semaphore.  Generic version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <semaphore.h>
#include <assert.h>

#include <hurdlock.h>

#include <pt-internal.h>

int
__sem_post (sem_t *sem)
{
  struct new_sem *isem = (struct new_sem *) sem;
  int flags = isem->pshared ? GSYNC_SHARED : 0;

#if __HAVE_64B_ATOMICS
  uint64_t d = atomic_load_relaxed (&isem->data);

  do
    {
      if ((d & SEM_VALUE_MASK) == SEM_VALUE_MAX)
	{
	  errno = EOVERFLOW;
	  return -1;
	}
    }
  while (!atomic_compare_exchange_weak_release (&isem->data, &d, d + 1));

  if ((d >> SEM_NWAITERS_SHIFT) != 0)
    /* Wake one waiter.  */
    __lll_wake (((unsigned int *) &isem->data) + SEM_VALUE_OFFSET, flags);
#else
  unsigned int v = atomic_load_relaxed (&isem->value);

  do
    {
      if ((v >> SEM_VALUE_SHIFT) == SEM_VALUE_MAX)
	{
	  errno = EOVERFLOW;
	  return -1;
	}
    }
  while (!atomic_compare_exchange_weak_release
	  (&isem->value, &v, v + (1 << SEM_VALUE_SHIFT)));

  if ((v & SEM_NWAITERS_MASK) != 0)
    /* Wake one waiter.  */
    __lll_wake (&isem->value, flags);
#endif

  return 0;
}
libpthread_hidden_def (__sem_post)
strong_alias (__sem_post, sem_post);
