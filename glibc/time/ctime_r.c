/* Return in BUF representation of time T in form of asctime
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

/* Return a string as returned by asctime which is the representation
   of *T in that form.  Reentrant version.  */
char *
__ctime64_r (const __time64_t *t, char *buf)
{
  struct tm tm;
  return __asctime_r (__localtime64_r (t, &tm), buf);
}

/* Provide a 32-bit variant if needed.  */

#if __TIMESIZE != 64

libc_hidden_def (__ctime64_r)

char *
ctime_r (const time_t *t, char *buf)
{
  __time64_t t64 = *t;
  return __ctime64_r (&t64, buf);
}

#endif
