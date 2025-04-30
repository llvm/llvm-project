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

#include "pthreadP.h"
#include <tls.h>
#include <libc-lock.h>

void
__libc_cleanup_push_defer (struct _pthread_cleanup_buffer *buffer)
{
  struct pthread *self = THREAD_SELF;

  buffer->__prev = THREAD_GETMEM (self, cleanup);

  /* Disable asynchronous cancellation for now.  */
  buffer->__canceltype = THREAD_GETMEM (self, canceltype);
  THREAD_SETMEM (self, canceltype, PTHREAD_CANCEL_DEFERRED);

  THREAD_SETMEM (self, cleanup, buffer);
}
libc_hidden_def (__libc_cleanup_push_defer)

void
__libc_cleanup_pop_restore (struct _pthread_cleanup_buffer *buffer)
{
  struct pthread *self = THREAD_SELF;

  THREAD_SETMEM (self, cleanup, buffer->__prev);

  THREAD_SETMEM (self, canceltype, buffer->__canceltype);
  if (buffer->__canceltype == PTHREAD_CANCEL_ASYNCHRONOUS)
      __pthread_testcancel ();
}
libc_hidden_def (__libc_cleanup_pop_restore)
