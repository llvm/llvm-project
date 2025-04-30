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
#include <time.h>
#include <sysdep.h>
#include "kernel-posix-timers.h"
#include <shlib-compat.h>

int
___timer_getoverrun (timer_t timerid)
{
  kernel_timer_t ktimerid = timerid_to_kernel_timer (timerid);
  return INLINE_SYSCALL_CALL (timer_getoverrun, ktimerid);
}
versioned_symbol (libc, ___timer_getoverrun, timer_getoverrun, GLIBC_2_34);
libc_hidden_ver (___timer_getoverrun, __timer_getoverrun)

#if TIMER_T_WAS_INT_COMPAT
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (librt, ___timer_getoverrun, timer_getoverrun, GLIBC_2_3_3);
# endif

# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_3_3)
int
__timer_getoverrun_old (int timerid)
{
  return __timer_getoverrun (__timer_compat_list[timerid]);
}
compat_symbol (librt, __timer_getoverrun_old, timer_getoverrun, GLIBC_2_2);
# endif /* OTHER_SHLIB_COMPAT */

#else /* !TIMER_T_WAS_INT_COMPAT */
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, ___timer_getoverrun, timer_getoverrun, GLIBC_2_2);
# endif
#endif /* !TIMER_T_WAS_INT_COMPAT */
