/* Sizes of structs with flexible array members.

   Copyright 2016-2021 Free Software Foundation, Inc.

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
   <https://www.gnu.org/licenses/>.

   Written by Paul Eggert.  */

#include <stddef.h>

/* Nonzero multiple of alignment of TYPE, suitable for FLEXSIZEOF below.
   On older platforms without _Alignof, use a pessimistic bound that is
   safe in practice even if FLEXIBLE_ARRAY_MEMBER is 1.
   On newer platforms, use _Alignof to get a tighter bound.  */

#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 201112
# define FLEXALIGNOF(type) (sizeof (type) & ~ (sizeof (type) - 1))
#else
# define FLEXALIGNOF(type) _Alignof (type)
#endif

/* Yield a properly aligned upper bound on the size of a struct of
   type TYPE with a flexible array member named MEMBER that is
   followed by N bytes of other data.  The result is suitable as an
   argument to malloc.  For example:

     struct s { int n; char d[FLEXIBLE_ARRAY_MEMBER]; };
     struct s *p = malloc (FLEXSIZEOF (struct s, d, n * sizeof (char)));

   FLEXSIZEOF (TYPE, MEMBER, N) is not simply (sizeof (TYPE) + N),
   since FLEXIBLE_ARRAY_MEMBER may be 1 on pre-C11 platforms.  Nor is
   it simply (offsetof (TYPE, MEMBER) + N), as that might yield a size
   that causes malloc to yield a pointer that is not properly aligned
   for TYPE; for example, if sizeof (int) == alignof (int) == 4,
   malloc (offsetof (struct s, d) + 3 * sizeof (char)) is equivalent
   to malloc (7) and might yield a pointer that is not a multiple of 4
   (which means the pointer is not properly aligned for struct s),
   whereas malloc (FLEXSIZEOF (struct s, d, 3 * sizeof (char))) is
   equivalent to malloc (8) and must yield a pointer that is a
   multiple of 4.

   Yield a value less than N if and only if arithmetic overflow occurs.  */

#define FLEXSIZEOF(type, member, n) \
   ((offsetof (type, member) + FLEXALIGNOF (type) - 1 + (n)) \
    & ~ (FLEXALIGNOF (type) - 1))
