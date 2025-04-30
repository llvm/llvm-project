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
___timer_settime64 (timer_t timerid, int flags,
                   const struct __itimerspec64 *value,
                   struct __itimerspec64 *ovalue)
{
  kernel_timer_t ktimerid = timerid_to_kernel_timer (timerid);

# ifdef __ASSUME_TIME64_SYSCALLS
#  ifndef __NR_timer_settime64
#   define __NR_timer_settime64 __NR_timer_settime
#  endif
  return INLINE_SYSCALL_CALL (timer_settime64, ktimerid, flags, value,
                              ovalue);
# else
#  ifdef __NR_timer_settime64
  int ret = INLINE_SYSCALL_CALL (timer_settime64, ktimerid, flags, value,
                                 ovalue);
  if (ret == 0 || errno != ENOSYS)
    return ret;
#  endif
  struct itimerspec its32, oits32;

  if (! in_time_t_range ((value->it_value).tv_sec)
      || ! in_time_t_range ((value->it_interval).tv_sec))
    {
      __set_errno (EOVERFLOW);
      return -1;
    }

  its32.it_interval = valid_timespec64_to_timespec (value->it_interval);
  its32.it_value = valid_timespec64_to_timespec (value->it_value);

  int retval = INLINE_SYSCALL_CALL (timer_settime, ktimerid, flags,
                                    &its32, ovalue ? &oits32 : NULL);
  if (retval == 0 && ovalue)
    {
      ovalue->it_interval = valid_timespec_to_timespec64 (oits32.it_interval);
      ovalue->it_value = valid_timespec_to_timespec64 (oits32.it_value);
    }

  return retval;
# endif
}

# if __TIMESIZE == 64
versioned_symbol (libc, ___timer_settime64, timer_settime, GLIBC_2_34);
#  if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, ___timer_settime64, timer_settime, GLIBC_2_2);
#  endif

#else /* __TIMESIZE != 64 */
libc_hidden_ver (___timer_settime64, __timer_settime64)
versioned_symbol (libc, ___timer_settime64, __timer_settime64, GLIBC_2_34);

int
__timer_settime (timer_t timerid, int flags, const struct itimerspec *value,
                 struct itimerspec *ovalue)
{
  struct __itimerspec64 its64, oits64;
  int retval;

  its64.it_interval = valid_timespec_to_timespec64 (value->it_interval);
  its64.it_value = valid_timespec_to_timespec64 (value->it_value);

  retval = __timer_settime64 (timerid, flags, &its64, ovalue ? &oits64 : NULL);
  if (retval == 0 && ovalue)
    {
      ovalue->it_interval = valid_timespec64_to_timespec (oits64.it_interval);
      ovalue->it_value = valid_timespec64_to_timespec (oits64.it_value);
    }

  return retval;
}
versioned_symbol (libc, __timer_settime, timer_settime, GLIBC_2_34);

#  if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, __timer_settime, timer_settime, GLIBC_2_2);
#  endif
# endif /* __TIMESIZE != 64 */

#else /* TIMER_T_WAS_INT_COMPAT */

extern __typeof (timer_settime) __timer_settime_new;
libc_hidden_proto (__timer_settime_new)

int
___timer_settime_new (timer_t timerid, int flags,
                      const struct itimerspec *value,
                      struct itimerspec *ovalue)
{
  kernel_timer_t ktimerid = timerid_to_kernel_timer (timerid);

  return INLINE_SYSCALL_CALL (timer_settime, ktimerid, flags, value, ovalue);
}
versioned_symbol (libc, ___timer_settime_new, timer_settime, GLIBC_2_34);
libc_hidden_ver (___timer_settime_new, __timer_settime_new)

# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (librt, ___timer_settime_new, timer_settime, GLIBC_2_3_3);
# endif

# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_3_3)
int
__timer_settime_old (int timerid, int flags, const struct itimerspec *value,
                     struct itimerspec *ovalue)
{
  return __timer_settime_new (__timer_compat_list[timerid], flags,
                              value, ovalue);
}
compat_symbol (librt, __timer_settime_old, timer_settime, GLIBC_2_2);
# endif

#endif /* TIMER_T_WAS_INT_COMPAT */
