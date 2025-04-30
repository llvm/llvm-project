/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sysdep.h>

/* Register a function to be called by exit.  */
int
__on_exit (void (*func) (int status, void *arg), void *arg)
{
  struct exit_function *new;

  /* As a QoI issue we detect NULL early with an assertion instead
     of a SIGSEGV at program exit when the handler is run (bug 20544).  */
  assert (func != NULL);

   __libc_lock_lock (__exit_funcs_lock);
  new = __new_exitfn (&__exit_funcs);

  if (new == NULL)
    {
      __libc_lock_unlock (__exit_funcs_lock);
      return -1;
    }

#ifdef PTR_MANGLE
  PTR_MANGLE (func);
#endif
  new->func.on.fn = func;
  new->func.on.arg = arg;
  new->flavor = ef_on;
  __libc_lock_unlock (__exit_funcs_lock);
  return 0;
}
weak_alias (__on_exit, on_exit)
