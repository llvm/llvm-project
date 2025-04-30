/* Convert `time_t' to `struct tm' in local time zone.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

/* The C Standard says that localtime and gmtime return the same pointer.  */
struct tm _tmbuf;


/* Return the `struct tm' representation of *T in local time,
   using *TP to store the result.  */
struct tm *
__localtime64_r (const __time64_t *t, struct tm *tp)
{
  return __tz_convert (*t, 1, tp);
}

/* Provide a 32-bit variant if needed.  */

#if __TIMESIZE != 64

struct tm *
__localtime_r (const time_t *t, struct tm *tp)
{
  __time64_t t64 = *t;
  return __localtime64_r (&t64, tp);
}
libc_hidden_def (__localtime64_r)

#endif

weak_alias (__localtime_r, localtime_r)


/* Return the `struct tm' representation of *T in local time.  */
struct tm *
__localtime64 (const __time64_t *t)
{
  return __tz_convert (*t, 1, &_tmbuf);
}
libc_hidden_def (__localtime64)

/* Provide a 32-bit variant if needed.  */

#if __TIMESIZE != 64

struct tm *
localtime (const time_t *t)
{
  __time64_t t64 = *t;
  return __localtime64 (&t64);
}
libc_hidden_def (localtime)

#endif
