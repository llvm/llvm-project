/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sysdep.h>
#include <signal.h>
#include <errno.h>
#include <shlib-compat.h>
#include "exit.h"

void
__new_quick_exit (int status)
{
  /* The new quick_exit, following C++11 18.5.12, does not run object
     destructors.   While C11 says nothing about object destructors,
     since it has none, the intent is to run the registered
     at_quick_exit handlers and then run _Exit immediately without
     disturbing the state of the process and threads.  */
  __run_exit_handlers (status, &__quick_exit_funcs, false, false);
}
versioned_symbol (libc, __new_quick_exit, quick_exit, GLIBC_2_24);

#if SHLIB_COMPAT(libc, GLIBC_2_10, GLIBC_2_24)
void
attribute_compat_text_section
__old_quick_exit (int status)
{
  /* The old quick_exit runs thread_local destructors.  */
  __run_exit_handlers (status, &__quick_exit_funcs, false, true);
}
compat_symbol (libc, __old_quick_exit, quick_exit, GLIBC_2_10);
#endif
