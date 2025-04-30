/* Concurrent allocation and initialization of a pointer.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <allocate_once.h>
#include <stdlib.h>
#include <stdbool.h>

void *
__libc_allocate_once_slow (void **place, void *(*allocate) (void *closure),
                           void (*deallocate) (void *closure, void *ptr),
                           void *closure)
{
  void *result = allocate (closure);
  if (result == NULL)
    return NULL;

  /* This loop implements a strong CAS on *place, with acquire-release
     MO semantics, from a weak CAS with relaxed-release MO.  */
  while (true)
    {
      /* Synchronizes with the acquire MO load in allocate_once.  */
      void *expected = NULL;
      if (atomic_compare_exchange_weak_release (place, &expected, result))
        return result;

      /* The failed CAS has relaxed MO semantics, so perform another
         acquire MO load.  */
      void *other_result = atomic_load_acquire (place);
      if (other_result == NULL)
        /* Spurious failure.  Try again.  */
        continue;

      /* We lost the race.  Free what we allocated and return the
         other result.  */
      if (deallocate == NULL)
        free (result);
      else
        deallocate (closure, result);
      return other_result;
    }

  return result;
}
libc_hidden_def (__libc_allocate_once_slow)
