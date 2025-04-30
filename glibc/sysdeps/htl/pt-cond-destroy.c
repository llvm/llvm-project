/* pthread_cond_destroy.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <pt-internal.h>

int
__pthread_cond_destroy (pthread_cond_t *cond)
{
  /* Set the wake request flag. */
  unsigned int wrefs = atomic_fetch_or_acquire (&cond->__wrefs, 1);

  __pthread_spin_wait (&cond->__lock);
  if (cond->__queue)
    {
      __pthread_spin_unlock (&cond->__lock);
      return EBUSY;
    }
  __pthread_spin_unlock (&cond->__lock);

  while (wrefs >> 1 != 0)
    {
      __gsync_wait (__mach_task_self (), (vm_offset_t) &cond->__wrefs, wrefs,
		  0, 0, 0);
      wrefs = atomic_load_acquire (&cond->__wrefs);
    }
  /* The memory the condvar occupies can now be reused.  */

  return 0;
}

weak_alias (__pthread_cond_destroy, pthread_cond_destroy);
