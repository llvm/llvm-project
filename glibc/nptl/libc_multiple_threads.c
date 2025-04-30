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

#include <pthreadP.h>

#if IS_IN (libc)
# ifndef TLS_MULTIPLE_THREADS_IN_TCB
/* Variable set to a nonzero value either if more than one thread runs or ran,
   or if a single-threaded process is trying to cancel itself.  See
   nptl/descr.h for more context on the single-threaded process case.  */
int __libc_multiple_threads;
libc_hidden_data_def (__libc_multiple_threads)
# endif
#endif
