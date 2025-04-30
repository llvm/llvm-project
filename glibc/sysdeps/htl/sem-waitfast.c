/* Lock a semaphore if it does not require blocking.  Generic version.
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
#include <errno.h>

#include <pt-internal.h>

int
__sem_waitfast (struct new_sem *isem, int definitive_result)
{
#if __HAVE_64B_ATOMICS
  uint64_t d = atomic_load_relaxed (&isem->data);

  do
    {
      if ((d & SEM_VALUE_MASK) == 0)
	break;
      if (atomic_compare_exchange_weak_acquire (&isem->data, &d, d - 1))
	/* Successful down.  */
	return 0;
    }
  while (definitive_result);
  return -1;
#else
  unsigned v = atomic_load_relaxed (&isem->value);

  do
    {
      if ((v >> SEM_VALUE_SHIFT) == 0)
	break;
      if (atomic_compare_exchange_weak_acquire (&isem->value,
	    &v, v - (1 << SEM_VALUE_SHIFT)))
	/* Successful down.  */
	return 0;
    }
  while (definitive_result);
  return -1;
#endif
}
