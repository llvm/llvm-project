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
#include <unwind-link.h>
#include <stdio.h>
#include <gnu/lib-names.h>

void
__pthread_exit (void *value)
{
  {
    struct unwind_link *unwind_link = __libc_unwind_link_get ();
    if (unwind_link == NULL)
      __libc_fatal (LIBGCC_S_SO
                    " must be installed for pthread_exit to work\n");
  }

  THREAD_SETMEM (THREAD_SELF, result, value);

  __do_cancel ();
}
libc_hidden_def (__pthread_exit)
weak_alias (__pthread_exit, pthread_exit)

/* Ensure that the unwinder is always linked in (the __pthread_unwind
   reference from __do_cancel is weak).  Use ___pthread_unwind_next
   (three underscores) to produce a strong reference to the same
   file.  */
PTHREAD_STATIC_FN_REQUIRE (___pthread_unwind_next)
