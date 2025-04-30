/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <sysdep.h>
#include <kernel-features.h>
#include "kernel-posix-timers.h"
#include <shlib-compat.h>

#if !TIMER_T_WAS_INT_COMPAT
int
___timer_gettime64 (timer_t timerid, struct __itimerspec64 *value)
{
  kernel_timer_t ktimerid = timerid_to_kernel_timer (timerid);

# ifndef __NR_timer_gettime64
#  define __NR_timer_gettime64 __NR_timer_gettime
# endif
  int ret = INLINE_SYSCALL_CALL (timer_gettime64, ktimerid, value);
# ifndef __ASSUME_TIME64_SYSCALLS
  if (ret == 0 || errno != ENOSYS)
    return ret;

  struct itimerspec its32;
  ret = INLINE_SYSCALL_CALL (timer_gettime, ktimerid, &its32);
  if (ret == 0)
    {
      value->it_interval = valid_timespec_to_timespec64 (its32.it_interval);
      value->it_value = valid_timespec_to_timespec64 (its32.it_value);
    }
# endif
  return ret;
}

# if __TIMESIZE == 64
versioned_symbol (libc, ___timer_gettime64, timer_gettime, GLIBC_2_34);
#  if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, ___timer_gettime64, timer_gettime, GLIBC_2_2);
#  endif

# else /* __TIMESIZE != 64 */
libc_hidden_ver (___timer_gettime64, __timer_gettime64)
versioned_symbol (libc, ___timer_gettime64, __timer_gettime64, GLIBC_2_34);

int
__timer_gettime (timer_t timerid, struct itimerspec *value)
{
  struct __itimerspec64 its64;
  int retval = __timer_gettime64 (timerid, &its64);
  if (retval == 0)
    {
      value->it_interval = valid_timespec64_to_timespec (its64.it_interval);
      value->it_value = valid_timespec64_to_timespec (its64.it_value);
    }

  return retval;
}
versioned_symbol (libc, __timer_gettime, timer_gettime, GLIBC_2_34);

#  if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, __timer_gettime, timer_gettime, GLIBC_2_2);
#  endif
# endif /* __TIMESIZE != 64 */

#else /* TIMER_T_WAS_INT_COMPAT */

extern __typeof (timer_gettime) __timer_gettime_new;
libc_hidden_proto (__timer_gettime_new)

int
___timer_gettime_new (timer_t timerid, struct itimerspec *value)
{
  kernel_timer_t ktimerid = timerid_to_kernel_timer (timerid);

  return INLINE_SYSCALL_CALL (timer_gettime, ktimerid, value);
}
versioned_symbol (libc, ___timer_gettime_new, timer_gettime, GLIBC_2_34);
libc_hidden_ver (___timer_gettime_new, __timer_gettime_new)

# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (librt, ___timer_gettime_new, timer_gettime, GLIBC_2_3_3);
# endif

# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_3_3)
int
__timer_gettime_old (int timerid, struct itimerspec *value)
{
  return __timer_gettime_new (__timer_compat_list[timerid], value);
}
compat_symbol (librt, __timer_gettime_old, timer_gettime, GLIBC_2_2);
# endif

#endif /* TIMER_T_WAS_INT_COMPAT */
