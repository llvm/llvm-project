/* Initialize a semaphore.  Generic version.
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
__sem_init (sem_t *sem, int pshared, unsigned value)
{
#ifdef SEM_VALUE_MAX
  if (value > SEM_VALUE_MAX)
    {
      errno = EINVAL;
      return -1;
    }
#endif

  struct new_sem *isem = (struct new_sem *) sem;

  *isem = (struct new_sem) __SEMAPHORE_INITIALIZER (value, pshared);
  return 0;
}

strong_alias (__sem_init, sem_init);
