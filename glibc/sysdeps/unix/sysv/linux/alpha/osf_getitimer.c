/* getitimer -- Get the state of an interval timer.  Linux/Alpha/tv32 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#include <time.h>
#include <sys/time.h>
#include <tv32-compat.h>

int
attribute_compat_text_section
__getitimer_tv32 (int which, struct __itimerval32 *curr_value)
{
  struct itimerval curr_value_64;
  if (__getitimer (which, &curr_value_64) == -1)
    return -1;

  /* Write all fields of 'curr_value' regardless of overflow.  */
  curr_value->it_interval
    = valid_timeval_to_timeval32 (curr_value_64.it_interval);
  curr_value->it_value
    = valid_timeval_to_timeval32 (curr_value_64.it_value);
  return 0;
}

compat_symbol (libc, __getitimer_tv32, getitimer, GLIBC_2_0);
#endif
