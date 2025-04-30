/* pthread_getspecific.  Hurd version.
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

void *
__pthread_getspecific (pthread_key_t key)
{
  struct __pthread *self;

  if (key < 0 || key >= __pthread_key_count
      || __pthread_key_destructors[key] == PTHREAD_KEY_INVALID)
    return NULL;

  self = _pthread_self ();
  if (key >= self->thread_specifics_size)
    return 0;

  return self->thread_specifics[key];
}
weak_alias (__pthread_getspecific, pthread_getspecific);
hidden_def (__pthread_getspecific)
