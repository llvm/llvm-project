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
#include <jmpbuf-unwind.h>

void
__pthread_cleanup_upto (__jmp_buf target, char *targetframe)
{
  struct pthread *self = THREAD_SELF;
  struct _pthread_cleanup_buffer *cbuf;

  /* Adjust all pointers used in comparisons, so that top of thread's
     stack is at the top of address space.  Without that, things break
     if stack is allocated above the main stack.  */
  uintptr_t adj = (uintptr_t) self->stackblock + self->stackblock_size;
  uintptr_t targetframe_adj = (uintptr_t) targetframe - adj;

  for (cbuf = THREAD_GETMEM (self, cleanup);
       cbuf != NULL && _JMPBUF_UNWINDS_ADJ (target, cbuf, adj);
       cbuf = cbuf->__prev)
    {
#if _STACK_GROWS_DOWN
      if ((uintptr_t) cbuf - adj <= targetframe_adj)
        {
          cbuf = NULL;
          break;
        }
#elif _STACK_GROWS_UP
      if ((uintptr_t) cbuf - adj >= targetframe_adj)
        {
          cbuf = NULL;
          break;
        }
#else
# error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
#endif

      /* Call the cleanup code.  */
      cbuf->__routine (cbuf->__arg);
    }

  THREAD_SETMEM (self, cleanup, cbuf);
}
libc_hidden_def (__pthread_cleanup_upto)
