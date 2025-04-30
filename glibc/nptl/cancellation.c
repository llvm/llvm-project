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

#include <setjmp.h>
#include <stdlib.h>
#include "pthreadP.h"
#include <futex-internal.h>


/* The next two functions are similar to pthread_setcanceltype() but
   more specialized for the use in the cancelable functions like write().
   They do not need to check parameters etc.  These functions must be
   AS-safe, with the exception of the actual cancellation, because they
   are called by wrappers around AS-safe functions like write().*/
int
__pthread_enable_asynccancel (void)
{
  struct pthread *self = THREAD_SELF;

  int oldval = THREAD_GETMEM (self, canceltype);
  THREAD_SETMEM (self, canceltype, PTHREAD_CANCEL_ASYNCHRONOUS);

  int ch = THREAD_GETMEM (self, cancelhandling);

  if (self->cancelstate == PTHREAD_CANCEL_ENABLE
      && (ch & CANCELED_BITMASK)
      && !(ch & EXITING_BITMASK)
      && !(ch & TERMINATED_BITMASK))
    {
      THREAD_SETMEM (self, result, PTHREAD_CANCELED);
      __do_cancel ();
    }

  return oldval;
}
libc_hidden_def (__pthread_enable_asynccancel)

/* See the comment for __pthread_enable_asynccancel regarding
   the AS-safety of this function.  */
void
__pthread_disable_asynccancel (int oldtype)
{
  /* If asynchronous cancellation was enabled before we do not have
     anything to do.  */
  if (oldtype == PTHREAD_CANCEL_ASYNCHRONOUS)
    return;

  struct pthread *self = THREAD_SELF;
  self->canceltype = PTHREAD_CANCEL_DEFERRED;
}
libc_hidden_def (__pthread_disable_asynccancel)
