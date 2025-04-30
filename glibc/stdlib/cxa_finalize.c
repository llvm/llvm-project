/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <stdlib.h>
#include "exit.h"
#include <register-atfork.h>
#include <sysdep.h>
#include <stdint.h>

/* If D is non-NULL, call all functions registered with `__cxa_atexit'
   with the same dso handle.  Otherwise, if D is NULL, call all of the
   registered handlers.  */
void
__cxa_finalize (void *d)
{
  struct exit_function_list *funcs;

  __libc_lock_lock (__exit_funcs_lock);

 restart:
  for (funcs = __exit_funcs; funcs; funcs = funcs->next)
    {
      struct exit_function *f;

      for (f = &funcs->fns[funcs->idx - 1]; f >= &funcs->fns[0]; --f)
	if ((d == NULL || d == f->func.cxa.dso_handle) && f->flavor == ef_cxa)
	  {
	    const uint64_t check = __new_exitfn_called;
	    void (*cxafn) (void *arg, int status) = f->func.cxa.fn;
	    void *cxaarg = f->func.cxa.arg;

	    /* We don't want to run this cleanup more than once.  The Itanium
	       C++ ABI requires that multiple calls to __cxa_finalize not
	       result in calling termination functions more than once.  One
	       potential scenario where that could happen is with a concurrent
	       dlclose and exit, where the running dlclose must at some point
	       release the list lock, an exiting thread may acquire it, and
	       without setting flavor to ef_free, might re-run this destructor
	       which could result in undefined behaviour.  Therefore we must
	       set flavor to ef_free to avoid calling this destructor again.
	       Note that the concurrent exit must also take the dynamic loader
	       lock (for library finalizer processing) and therefore will
	       block while dlclose completes the processing of any in-progress
	       exit functions. Lastly, once we release the list lock for the
	       entry marked ef_free, we must not read from that entry again
	       since it may have been reused by the time we take the list lock
	       again.  Lastly the detection of new registered exit functions is
	       based on a monotonically incrementing counter, and there is an
	       ABA if between the unlock to run the exit function and the
	       re-lock after completion the user registers 2^64 exit functions,
	       the implementation will not detect this and continue without
	       executing any more functions.

	       One minor issue remains: A registered exit function that is in
	       progress by a call to dlclose() may not completely finish before
	       the next registered exit function is run. This may, according to
	       some readings of POSIX violate the requirement that functions
	       run in effective LIFO order.  This should probably be fixed in a
	       future implementation to ensure the functions do not run in
	       parallel.  */
	    f->flavor = ef_free;

#ifdef PTR_DEMANGLE
	    PTR_DEMANGLE (cxafn);
#endif
	    /* Unlock the list while we call a foreign function.  */
	    __libc_lock_unlock (__exit_funcs_lock);
	    cxafn (cxaarg, 0);
	    __libc_lock_lock (__exit_funcs_lock);

	    /* It is possible that that last exit function registered
	       more exit functions.  Start the loop over.  */
	    if (__glibc_unlikely (check != __new_exitfn_called))
	      goto restart;
	  }
    }

  /* Also remove the quick_exit handlers, but do not call them.  */
  for (funcs = __quick_exit_funcs; funcs; funcs = funcs->next)
    {
      struct exit_function *f;

      for (f = &funcs->fns[funcs->idx - 1]; f >= &funcs->fns[0]; --f)
	if (d == NULL || d == f->func.cxa.dso_handle)
	  f->flavor = ef_free;
    }

  /* Remove the registered fork handlers.  We do not have to
     unregister anything if the program is going to terminate anyway.  */
  if (d != NULL)
    UNREGISTER_ATFORK (d);
  __libc_lock_unlock (__exit_funcs_lock);
}
