/* Private libc-internal interface for mutex locks.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#ifndef _HTL_LIBC_LOCK_H
#define _HTL_LIBC_LOCK_H 1

#include_next <libc-lock.h>
#include <bits/cancelation.h>

#undef __libc_cleanup_region_start
#undef __libc_cleanup_region_end
#undef __libc_cleanup_end
#undef __libc_cleanup_push
#undef __libc_cleanup_pop

#define __libc_cleanup_region_start(DOIT, FCT, ARG) \
  {									      \
    struct __pthread_cancelation_handler **__handlers = NULL;		      \
    struct __pthread_cancelation_handler __handler;			      \
    int __registered = 0;						      \
    if (DOIT)								      \
      {									      \
	__handler.__handler = FCT;					      \
	__handler.__arg = ARG;						      \
	if (__pthread_get_cleanup_stack != NULL)			      \
	  {								      \
	    __handlers = __pthread_get_cleanup_stack ();		      \
	    __handler.__next = *__handlers;				      \
	    *__handlers = &__handler;					      \
	    __registered = 1;						      \
	  }								      \
      }									      \

#define __libc_cleanup_end(DOIT) \
    if (__registered)							      \
      *__handlers = __handler.__next;					      \
    if (DOIT)								      \
      __handler.__handler (__handler.__arg);				      \

#define __libc_cleanup_region_end(DOIT) \
    __libc_cleanup_end(DOIT)						      \
  }

#define __libc_cleanup_push(fct, arg) __libc_cleanup_region_start (1, fct, arg)
#define __libc_cleanup_pop(execute) __libc_cleanup_region_end (execute)

#if !IS_IN (libpthread)
# ifdef weak_extern
weak_extern (__pthread_get_cleanup_stack)
# else
#  pragma weak __pthread_get_cleanup_stack
# endif
#endif

#endif
