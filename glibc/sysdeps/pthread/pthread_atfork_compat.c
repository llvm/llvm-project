/* Compat pthread_atfork implementation.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <pthread_atfork_compat.h>
#include <shlib-compat.h>

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_3)
# define __pthread_atfork __dyn_pthread_atfork
# include "pthread_atfork.c"
# undef __pthread_atfork
compat_symbol (libpthread, __dyn_pthread_atfork, pthread_atfork,
	       PTHREAD_ATFORK_COMPAT_INTRODUCED);
#endif
