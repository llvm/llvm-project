/* libc initialization for libpthread.  Hurd pthread version.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>
#include <pthread-functions.h>

__thread struct __pthread_cancelation_handler *__pthread_cleanup_stack;

void
__libc_pthread_init (const struct pthread_functions *functions)
{
#ifdef SHARED
  /* We copy the content of the variable pointed to by the FUNCTIONS
     parameter to one in libc.so since this means access to the array
     can be done with one memory access instead of two.  */
  memcpy (&__libc_pthread_functions, functions,
	  sizeof (__libc_pthread_functions));
  __libc_pthread_functions_init = 1;
#endif
}
