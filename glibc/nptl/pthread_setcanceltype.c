/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <errno.h>
#include "pthreadP.h"
#include <atomic.h>


int
__pthread_setcanceltype (int type, int *oldtype)
{
  if (type < PTHREAD_CANCEL_DEFERRED || type > PTHREAD_CANCEL_ASYNCHRONOUS)
    return EINVAL;

  volatile struct pthread *self = THREAD_SELF;

  if (oldtype != NULL)
    *oldtype = self->canceltype;
  self->canceltype = type;
  if (type == PTHREAD_CANCEL_ASYNCHRONOUS)
    __pthread_testcancel ();

  return 0;
}
libc_hidden_def (__pthread_setcanceltype)
weak_alias (__pthread_setcanceltype, pthread_setcanceltype)
