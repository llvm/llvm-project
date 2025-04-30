/* Check stack alignment.  PowerPC version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#define TEST_STACK_ALIGN_INIT() \
  ({									     \
    /* Altivec __vector int etc. needs 16byte aligned stack.		     \
       Instead of using altivec.h here, use aligned attribute instead.  */   \
    struct _S								     \
      {									     \
        int _i __attribute__((aligned (16)));				     \
	int _j[3];							     \
      } _s = { ._i = 18, ._j[0] = 19, ._j[1] = 20, ._j[2] = 21 };	     \
    printf ("__vector int:  { %d, %d, %d, %d } %p %zu\n", _s._i, _s._j[0],   \
            _s._j[1], _s._j[2], &_s, __alignof (_s));			     \
    is_aligned (&_s, __alignof (_s));					     \
   })

#include_next <tst-stack-align.h>
