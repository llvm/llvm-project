/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <time.h>

/* Return a string as returned by asctime which
   is the representation of *T in that form.  */
char *
__ctime64 (const __time64_t *t)
{
  /* The C Standard says ctime (t) is equivalent to asctime (localtime (t)).
     In particular, ctime and asctime must yield the same pointer.  */
  return asctime (__localtime64 (t));
}

/* Provide a 32-bit variant if needed.  */

#if __TIMESIZE != 64

libc_hidden_def (__ctime64)

char *
ctime (const time_t *t)
{
  __time64_t t64 = *t;
  return __ctime64 (&t64);
}

#endif
