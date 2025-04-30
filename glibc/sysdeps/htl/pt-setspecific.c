/* pthread_setspecific.  Generic version.
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
__pthread_setspecific (pthread_key_t key, const void *value)
{
  struct __pthread *self = _pthread_self ();

  if (key < 0 || key >= __pthread_key_count
      || __pthread_key_destructors[key] == PTHREAD_KEY_INVALID)
    return EINVAL;

  if (key >= self->thread_specifics_size)
    {
      /* Amortize reallocation cost.  */
      int newsize = 2 * key + 1;
      void **new = realloc (self->thread_specifics,
			    newsize * sizeof (new[0]));
      if (new == NULL)
	return ENOMEM;

      memset (&new[self->thread_specifics_size], 0,
	      (newsize - self->thread_specifics_size) * sizeof (new[0]));
      self->thread_specifics = new;
      self->thread_specifics_size = newsize;
    }

  self->thread_specifics[key] = (void *) value;
  return 0;
}
weak_alias (__pthread_setspecific, pthread_setspecific);
hidden_def (__pthread_setspecific)
