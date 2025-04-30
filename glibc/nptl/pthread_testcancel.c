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
___pthread_testcancel (void)
{
  struct pthread *self = THREAD_SELF;
  int cancelhandling = THREAD_GETMEM (self, cancelhandling);
  if (self->cancelstate == PTHREAD_CANCEL_ENABLE
      && (cancelhandling & CANCELED_BITMASK)
      && !(cancelhandling & EXITING_BITMASK)
      && !(cancelhandling & TERMINATED_BITMASK))
    {
      THREAD_SETMEM (self, result, PTHREAD_CANCELED);
      __do_cancel ();
    }
}
versioned_symbol (libc, ___pthread_testcancel, pthread_testcancel, GLIBC_2_34);
libc_hidden_ver (___pthread_testcancel, __pthread_testcancel)
#ifndef SHARED
strong_alias (___pthread_testcancel, __pthread_testcancel)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libc, ___pthread_testcancel, pthread_testcancel, GLIBC_2_0);
#endif
