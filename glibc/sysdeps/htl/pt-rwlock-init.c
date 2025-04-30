/* Initialize a rwlock.  Generic version.
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
#include <string.h>
#include <pt-internal.h>

int
_pthread_rwlock_init (pthread_rwlock_t *rwlock,
		      const pthread_rwlockattr_t *attr)
{
  ASSERT_TYPE_SIZE (pthread_rwlock_t, __SIZEOF_PTHREAD_RWLOCK_T);

  *rwlock = (pthread_rwlock_t) __PTHREAD_RWLOCK_INITIALIZER;

  if (attr == NULL
      || memcmp (attr, &__pthread_default_rwlockattr, sizeof (*attr)) == 0)
    /* Use the default attributes.  */
    return 0;

  /* Non-default attributes.  */

  rwlock->__attr = malloc (sizeof *attr);
  if (rwlock->__attr == NULL)
    return ENOMEM;

  *rwlock->__attr = *attr;
  return 0;
}

strong_alias (_pthread_rwlock_init, pthread_rwlock_init);
