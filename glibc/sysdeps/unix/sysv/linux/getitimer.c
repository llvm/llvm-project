/* getitimer -- Get the state of an interval timer.  Linux/32 version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sysdep.h>
#include <tv32-compat.h>

int
__getitimer64 (__itimer_which_t which, struct __itimerval64 *curr_value)
{
#if __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64
  return INLINE_SYSCALL_CALL (getitimer, which, curr_value);
#else
  struct __itimerval32 curr_value_32;

  if (INLINE_SYSCALL_CALL (getitimer, which, &curr_value_32) == -1)
    return -1;

  curr_value->it_interval
    = valid_timeval32_to_timeval64 (curr_value_32.it_interval);
  curr_value->it_value
    = valid_timeval32_to_timeval64 (curr_value_32.it_value);
  return 0;
#endif
}

#if __TIMESIZE != 64
libc_hidden_def (__getitimer64)
int
__getitimer (__itimer_which_t which, struct itimerval *curr_value)
{
  struct __itimerval64 val64;
  if (__getitimer64 (which, &val64) != 0)
    return -1;

  curr_value->it_interval
    = valid_timeval64_to_timeval (val64.it_interval);
  curr_value->it_value
    = valid_timeval64_to_timeval (val64.it_value);

  return 0;
}
#endif
weak_alias (__getitimer, getitimer)
