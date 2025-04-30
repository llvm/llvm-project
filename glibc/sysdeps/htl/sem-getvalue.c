/* Get the value of a semaphore.  Generic version.
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
#include <pt-internal.h>

int
__sem_getvalue (sem_t *restrict sem, int *restrict value)
{
  struct new_sem *isem = (struct new_sem *) sem;

#if __HAVE_64B_ATOMICS
  *value = atomic_load_relaxed (&isem->data) & SEM_VALUE_MASK;
#else
  *value = atomic_load_relaxed (&isem->value) >> SEM_VALUE_SHIFT;
#endif

  return 0;
}

strong_alias (__sem_getvalue, sem_getvalue);
