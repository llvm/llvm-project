/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <atomic.h>
#include <stdlib.h>
#include <set-hooks.h>
#include <libc-internal.h>
#include <unwind-link.h>
#include <dlfcn/dlerror.h>

#include "../nss/nsswitch.h"
#include "../libio/libioP.h"

DEFINE_HOOK (__libc_subfreeres, (void));

symbol_set_define (__libc_freeres_ptrs);

extern void __libpthread_freeres (void)
#if PTHREAD_IN_LIBC && defined SHARED
/* It is possible to call __libpthread_freeres directly in shared
   builds with an integrated libpthread.  */
  attribute_hidden
#else
  __attribute__ ((weak))
#endif
  ;

void __libc_freeres_fn_section
__libc_freeres (void)
{
  /* This function might be called from different places.  So better
     protect for multiple executions since these are fatal.  */
  static long int already_called;

  if (!atomic_compare_and_exchange_bool_acq (&already_called, 1, 0))
    {
      void *const *p;

      call_function_static_weak (__nss_module_freeres);
      call_function_static_weak (__nss_action_freeres);
      call_function_static_weak (__nss_database_freeres);

      _IO_cleanup ();

      /* We run the resource freeing after IO cleanup.  */
      RUN_HOOK (__libc_subfreeres, ());

      call_function_static_weak (__libpthread_freeres);

#ifdef SHARED
      __libc_unwind_link_freeres ();
#endif

      call_function_static_weak (__libc_dlerror_result_free);

      for (p = symbol_set_first_element (__libc_freeres_ptrs);
           !symbol_set_end_p (__libc_freeres_ptrs, p); ++p)
        free (*p);
    }
}
libc_hidden_def (__libc_freeres)
