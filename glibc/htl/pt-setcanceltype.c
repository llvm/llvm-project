/* Set the cancel type for the calling thread.
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
__pthread_setcanceltype (int type, int *oldtype)
{
  struct __pthread *p = _pthread_self ();

  switch (type)
    {
    default:
      return EINVAL;
    case PTHREAD_CANCEL_DEFERRED:
    case PTHREAD_CANCEL_ASYNCHRONOUS:
      break;
    }

  __pthread_mutex_lock (&p->cancel_lock);
  if (oldtype != NULL)
    *oldtype = p->cancel_type;
  p->cancel_type = type;
  __pthread_mutex_unlock (&p->cancel_lock);

  return 0;
}

weak_alias (__pthread_setcanceltype, pthread_setcanceltype);
