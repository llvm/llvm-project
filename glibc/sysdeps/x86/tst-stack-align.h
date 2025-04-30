/* Check stack alignment.  X86 version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

typedef struct { int i[16]; } int_al16 __attribute__((aligned (16)));

#define TEST_STACK_ALIGN_INIT() \
  ({								\
    int_al16 _m;						\
    printf ("int_al16:  %p %zu\n", &_m, __alignof (int_al16));	\
    is_aligned (&_m, __alignof (int_al16));			\
   })

#include_next <tst-stack-align.h>
