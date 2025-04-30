/* Cancelation.  Generic version.
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

#ifndef _BITS_CANCELATION_H
#define _BITS_CANCELATION_H	1

struct __pthread_cancelation_handler
{
  void (*__handler) (void *);
  void *__arg;
  struct __pthread_cancelation_handler *__next;
};

/* Returns the thread local location of the cleanup handler stack.  */
struct __pthread_cancelation_handler **__pthread_get_cleanup_stack (void);

#define __pthread_cleanup_push(rt, rtarg) \
	{ \
	  struct __pthread_cancelation_handler **__handlers \
	    = __pthread_get_cleanup_stack (); \
	  struct __pthread_cancelation_handler __handler = \
	    { \
	      (rt), \
	      (rtarg), \
	      *__handlers \
	    }; \
	  *__handlers = &__handler;

#define __pthread_cleanup_pop(execute) \
	  if (execute) \
	    __handler.__handler (__handler.__arg); \
	  *__handlers = __handler.__next; \
	}

#endif /* _BITS_CANCELATION_H */
