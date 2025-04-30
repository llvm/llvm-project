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

#include <stdlib.h>
#include "pthreadP.h"
#include <shlib-compat.h>

void
__cleanup_fct_attribute
___pthread_register_cancel_defer (__pthread_unwind_buf_t *buf)
{
  struct pthread_unwind_buf *ibuf = (struct pthread_unwind_buf *) buf;
  struct pthread *self = THREAD_SELF;

  /* Store old info.  */
  ibuf->priv.data.prev = THREAD_GETMEM (self, cleanup_jmp_buf);
  ibuf->priv.data.cleanup = THREAD_GETMEM (self, cleanup);

  /* Disable asynchronous cancellation for now.  */
  ibuf->priv.data.canceltype = THREAD_GETMEM (self, canceltype);
  THREAD_SETMEM (self, canceltype, PTHREAD_CANCEL_DEFERRED);

  /* Store the new cleanup handler info.  */
  THREAD_SETMEM (self, cleanup_jmp_buf, (struct pthread_unwind_buf *) buf);
}
versioned_symbol (libc, ___pthread_register_cancel_defer,
		  __pthread_register_cancel_defer, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_register_cancel_defer,
	       __pthread_register_cancel_defer, GLIBC_2_3_3);
#endif

void
__cleanup_fct_attribute
___pthread_unregister_cancel_restore (__pthread_unwind_buf_t *buf)
{
  struct pthread *self = THREAD_SELF;
  struct pthread_unwind_buf *ibuf = (struct pthread_unwind_buf *) buf;

  THREAD_SETMEM (self, cleanup_jmp_buf, ibuf->priv.data.prev);

  THREAD_SETMEM (self, canceltype, ibuf->priv.data.canceltype);
  if (ibuf->priv.data.canceltype == PTHREAD_CANCEL_ASYNCHRONOUS)
    __pthread_testcancel ();
}
versioned_symbol (libc, ___pthread_unregister_cancel_restore,
		  __pthread_unregister_cancel_restore, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_unregister_cancel_restore,
	       __pthread_unregister_cancel_restore, GLIBC_2_3_3);
#endif
