/* Check stack alignment.  Generic version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <stdint.h>

int
__attribute__ ((weak, noclone, noinline))
is_aligned (void *p, int align)
{
  return (((uintptr_t) p) & (align - 1)) != 0;
}

#ifndef TEST_STACK_ALIGN_INIT
# define TEST_STACK_ALIGN_INIT() 0
#endif

#define TEST_STACK_ALIGN() \
  ({								     \
    double _d = 12.0;						     \
    long double _ld = 15.0;					     \
    int _ret = TEST_STACK_ALIGN_INIT ();			     \
								     \
    printf ("double:  %g %p %zu\n", _d, &_d, __alignof (double));    \
    _ret += is_aligned (&_d, __alignof (double));		     \
								     \
    printf ("ldouble: %Lg %p %zu\n", _ld, &_ld,			     \
	    __alignof (long double));				     \
    _ret += is_aligned (&_ld, __alignof (long double));		     \
    _ret;							     \
   })
