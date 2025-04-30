/* pthread_attr_setstack.  Generic version.
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

#include <pt-internal.h>
#include <pthreadP.h>

int
__pthread_attr_setstack (pthread_attr_t *attr, void *stackaddr, size_t stacksize)
{
  int err;
  size_t s;

  /* pthread_attr_setstack should always succeed, thus we set the size
     first as it is more discriminating.  */
  __pthread_attr_getstacksize (attr, &s);

  err = __pthread_attr_setstacksize (attr, stacksize);
  if (err)
    return err;

  err = __pthread_attr_setstackaddr (attr, stackaddr);
  if (err)
    {
      int e = __pthread_attr_setstacksize (attr, s);
      assert_perror (e);

      return err;
    }

  return 0;
}
weak_alias (__pthread_attr_setstack, pthread_attr_setstack)
