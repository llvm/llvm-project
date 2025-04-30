/* clock_getres -- Get the resolution of a POSIX clockid_t.  Linux version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <errno.h>
#include <time.h>

#include <sysdep-vdso.h>
#include <shlib-compat.h>
#include <kernel-features.h>

/* Get resolution of clock.  */
int
__clock_getres64 (clockid_t clock_id, struct __timespec64 *res)
{
  int r;

#ifndef __NR_clock_getres_time64
# define __NR_clock_getres_time64 __NR_clock_getres
#endif

#ifdef HAVE_CLOCK_GETRES64_VSYSCALL
  r = INLINE_VSYSCALL (clock_getres_time64, 2, clock_id, res);
#else
  r = INLINE_SYSCALL_CALL (clock_getres_time64, clock_id, res);
#endif
  if (r == 0 || errno != ENOSYS)
    return r;

#ifndef __ASSUME_TIME64_SYSCALLS
  /* Fallback code that uses 32-bit support.  */
  struct timespec ts32;
# ifdef HAVE_CLOCK_GETRES_VSYSCALL
  r = INLINE_VSYSCALL (clock_getres, 2, clock_id, &ts32);
# else
  r = INLINE_SYSCALL_CALL (clock_getres, clock_id, &ts32);
# endif
  if (r == 0 && res != NULL)
    *res = valid_timespec_to_timespec64 (ts32);
#endif

  return r;
}

#if __TIMESIZE != 64
libc_hidden_def (__clock_getres64)

int
__clock_getres (clockid_t clock_id, struct timespec *res)
{
  struct __timespec64 ts64;
  int retval;

  retval = __clock_getres64 (clock_id, &ts64);
  if (retval == 0 && res != NULL)
    *res = valid_timespec64_to_timespec (ts64);

  return retval;
}
#endif
libc_hidden_def (__clock_getres)

versioned_symbol (libc, __clock_getres, clock_getres, GLIBC_2_17);
/* clock_getres moved to libc in version 2.17;
   old binaries may expect the symbol version it had in librt.  */
#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
strong_alias (__clock_getres, __clock_getres_2);
compat_symbol (libc, __clock_getres_2, clock_getres, GLIBC_2_2);
#endif
