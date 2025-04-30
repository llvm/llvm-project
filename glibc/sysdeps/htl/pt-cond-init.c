/* pthread_cond_init.  Generic version.
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
#include <assert.h>
#include <string.h>

#include <pt-internal.h>

int
__pthread_cond_init (pthread_cond_t *cond, const pthread_condattr_t * attr)
{
  ASSERT_TYPE_SIZE (pthread_cond_t, __SIZEOF_PTHREAD_COND_T);

  *cond = (pthread_cond_t) __PTHREAD_COND_INITIALIZER;

  if (attr == NULL
      || memcmp (attr, &__pthread_default_condattr, sizeof (*attr)) == 0)
    /* Use the default attributes.  */
    return 0;

  /* Non-default attributes.  */

  cond->__attr = malloc (sizeof *attr);
  if (cond->__attr == NULL)
    return ENOMEM;

  *cond->__attr = *attr;
  return 0;
}

weak_alias (__pthread_cond_init, pthread_cond_init);
