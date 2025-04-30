/* Add an explicit cancelation point.
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
#include <pthreadP.h>

void
__pthread_testcancel (void)
{
  struct __pthread *p = _pthread_self ();
  int cancelled;

  __pthread_mutex_lock (&p->cancel_lock);
  cancelled = (p->cancel_state == PTHREAD_CANCEL_ENABLE) && p->cancel_pending;
  __pthread_mutex_unlock (&p->cancel_lock);

  if (cancelled)
    __pthread_exit (PTHREAD_CANCELED);
}
strong_alias (__pthread_testcancel, pthread_testcancel)
