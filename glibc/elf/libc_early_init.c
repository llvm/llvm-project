/* Early initialization of libc.so, libc.so side.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <ctype.h>
#include <elision-conf.h>
#include <libc-early-init.h>
#include <libc-internal.h>
#include <lowlevellock.h>
#include <pthread_early_init.h>
#include <sys/single_threaded.h>

#ifdef SHARED
_Bool __libc_initial;
#endif

void
__libc_early_init (_Bool initial)
{
  /* Initialize ctype data.  */
  __ctype_init ();

  /* Only the outer namespace is marked as single-threaded.  */
  __libc_single_threaded = initial;

#ifdef SHARED
  __libc_initial = initial;
#endif

  __pthread_early_init ();

#if ENABLE_ELISION_SUPPORT
  __lll_elision_init ();
#endif
}
