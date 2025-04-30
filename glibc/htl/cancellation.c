/* Set the cancel type during blocking calls.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <pthreadP.h>
#include <pt-internal.h>

int __pthread_enable_asynccancel (void)
{
  struct __pthread *p = _pthread_self ();
  int oldtype;

  __pthread_mutex_lock (&p->cancel_lock);
  oldtype = p->cancel_type;
  p->cancel_type = PTHREAD_CANCEL_ASYNCHRONOUS;
  __pthread_mutex_unlock (&p->cancel_lock);

  __pthread_testcancel ();

  return oldtype;
}

void __pthread_disable_asynccancel (int oldtype)
{
  struct __pthread *p = _pthread_self ();

  __pthread_mutex_lock (&p->cancel_lock);
  p->cancel_type = oldtype;
  __pthread_mutex_unlock (&p->cancel_lock);
}
