/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>
   and Richard Henderson <rth@redhat.com>, 2003.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "pthreadP.h"
#include <libc-diag.h>
#include <jmpbuf-unwind.h>
#include <shlib-compat.h>

#ifdef _STACK_GROWS_DOWN
# define FRAME_LEFT(frame, other, adj) \
  ((uintptr_t) frame - adj >= (uintptr_t) other - adj)
#elif _STACK_GROWS_UP
# define FRAME_LEFT(frame, other, adj) \
  ((uintptr_t) frame - adj <= (uintptr_t) other - adj)
#else
# error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
#endif

static _Unwind_Reason_Code
unwind_stop (int version, _Unwind_Action actions,
	     _Unwind_Exception_Class exc_class,
	     struct _Unwind_Exception *exc_obj,
	     struct _Unwind_Context *context, void *stop_parameter)
{
  struct pthread_unwind_buf *buf = stop_parameter;
  struct pthread *self = THREAD_SELF;
  struct _pthread_cleanup_buffer *curp = THREAD_GETMEM (self, cleanup);
  int do_longjump = 0;

  /* Adjust all pointers used in comparisons, so that top of thread's
     stack is at the top of address space.  Without that, things break
     if stack is allocated above the main stack.  */
  uintptr_t adj = (uintptr_t) self->stackblock + self->stackblock_size;

  /* Do longjmp if we're at "end of stack", aka "end of unwind data".
     We assume there are only C frame without unwind data in between
     here and the jmp_buf target.  Otherwise simply note that the CFA
     of a function is NOT within it's stack frame; it's the SP of the
     previous frame.  */
  if ((actions & _UA_END_OF_STACK)
      || ! _JMPBUF_CFA_UNWINDS_ADJ (buf->cancel_jmp_buf[0].jmp_buf, context,
				    adj))
    do_longjump = 1;

  if (__glibc_unlikely (curp != NULL))
    {
      /* Handle the compatibility stuff.  Execute all handlers
	 registered with the old method which would be unwound by this
	 step.  */
      struct _pthread_cleanup_buffer *oldp = buf->priv.data.cleanup;
      void *cfa = (void *) (_Unwind_Ptr) _Unwind_GetCFA (context);

      if (curp != oldp && (do_longjump || FRAME_LEFT (cfa, curp, adj)))
	{
	  do
	    {
	      /* Pointer to the next element.  */
	      struct _pthread_cleanup_buffer *nextp = curp->__prev;

	      /* Call the handler.  */
	      curp->__routine (curp->__arg);

	      /* To the next.  */
	      curp = nextp;
	    }
	  while (curp != oldp
		 && (do_longjump || FRAME_LEFT (cfa, curp, adj)));

	  /* Mark the current element as handled.  */
	  THREAD_SETMEM (self, cleanup, curp);
	}
    }

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* This call results in a -Wstringop-overflow warning because struct
     pthread_unwind_buf is smaller than jmp_buf.  setjmp and longjmp
     do not use anything beyond the common prefix (they never access
     the saved signal mask), so that is a false positive.  */
  DIAG_IGNORE_NEEDS_COMMENT (11, "-Wstringop-overflow=");
#endif
  if (do_longjump)
    __libc_unwind_longjmp ((struct __jmp_buf_tag *) buf->cancel_jmp_buf, 1);
  DIAG_POP_NEEDS_COMMENT;

  return _URC_NO_REASON;
}


static void
unwind_cleanup (_Unwind_Reason_Code reason, struct _Unwind_Exception *exc)
{
  /* When we get here a C++ catch block didn't rethrow the object.  We
     cannot handle this case and therefore abort.  */
  __libc_fatal ("FATAL: exception not rethrown\n");
}


void
__cleanup_fct_attribute __attribute ((noreturn))
__pthread_unwind (__pthread_unwind_buf_t *buf)
{
  struct pthread_unwind_buf *ibuf = (struct pthread_unwind_buf *) buf;
  struct pthread *self = THREAD_SELF;

  /* This is not a catchable exception, so don't provide any details about
     the exception type.  We do need to initialize the field though.  */
  THREAD_SETMEM (self, exc.exception_class, 0);
  THREAD_SETMEM (self, exc.exception_cleanup, &unwind_cleanup);

  _Unwind_ForcedUnwind (&self->exc, unwind_stop, ibuf);
  /* NOTREACHED */

  /* We better do not get here.  */
  abort ();
}
libc_hidden_def (__pthread_unwind)

void
__cleanup_fct_attribute __attribute ((noreturn))
___pthread_unwind_next (__pthread_unwind_buf_t *buf)
{
  struct pthread_unwind_buf *ibuf = (struct pthread_unwind_buf *) buf;

  __pthread_unwind ((__pthread_unwind_buf_t *) ibuf->priv.data.prev);
}
versioned_symbol (libc, ___pthread_unwind_next, __pthread_unwind_next,
		  GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_unwind_next, __pthread_unwind_next,
	       GLIBC_2_3_3);
#endif
