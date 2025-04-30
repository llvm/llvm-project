/* Free resources stored in thread-local variables on thread exit.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <dlfcn/dlerror.h>
#include <libc-internal.h>
#include <malloc-internal.h>
#include <resolv/resolv-internal.h>
#include <rpc/rpc.h>
#include <string.h>
#include <tls-internal.h>
#include <shlib-compat.h>

/* Thread shutdown function.  Note that this function must be called
   for threads during shutdown for correctness reasons.  Unlike
   __libc_subfreeres, skipping calls to it is not a valid optimization.
   This is called directly from pthread_create as the thread exits.  */
void
__libc_thread_freeres (void)
{
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_32)
  __rpc_thread_destroy ();
#endif
  call_function_static_weak (__res_thread_freeres);
  __glibc_tls_internal_free ();
  call_function_static_weak (__libc_dlerror_result_free);

  /* This should come last because it shuts down malloc for this
     thread and the other shutdown functions might well call free.  */
  call_function_static_weak (__malloc_arena_thread_freeres);
}
