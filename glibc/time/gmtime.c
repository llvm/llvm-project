/* Convert `time_t' to `struct tm' in UTC.
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

/* Return the `struct tm' representation of *T in UTC,
   using *TP to store the result.  */
struct tm *
__gmtime64_r (const __time64_t *t, struct tm *tp)
{
  return __tz_convert (*t, 0, tp);
}

/* Provide a 32-bit variant if needed.  */

#if __TIMESIZE != 64

libc_hidden_def (__gmtime64_r)

struct tm *
__gmtime_r (const time_t *t, struct tm *tp)
{
  __time64_t t64 = *t;
  return __gmtime64_r (&t64, tp);
}

#endif

libc_hidden_def (__gmtime_r)
weak_alias (__gmtime_r, gmtime_r)

/* Return the `struct tm' representation of *T in UTC.  */
struct tm *
__gmtime64 (const __time64_t *t)
{
  return __tz_convert (*t, 0, &_tmbuf);
}

/* Provide a 32-bit variant if needed.  */

#if __TIMESIZE != 64

libc_hidden_def (__gmtime64)

struct tm *
gmtime (const time_t *t)
{
  __time64_t t64 = *t;
  return __gmtime64 (&t64);
}

#endif
