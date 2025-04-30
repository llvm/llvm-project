/* gettimeofday - set time.  Linux version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <time.h>
#include <string.h>

/* Optimize the function call by setting the PLT directly to vDSO symbol.  */
#ifdef USE_IFUNC_GETTIMEOFDAY
# include <sysdep.h>
# include <sysdep-vdso.h>

# ifdef SHARED
# include <dl-vdso.h>
# include <libc-vdso.h>

static int
__gettimeofday_syscall (struct timeval *restrict tv, void *restrict tz)
{
  if (__glibc_unlikely (tz != 0))
    memset (tz, 0, sizeof *tz);
  return INLINE_SYSCALL_CALL (gettimeofday, tv, tz);
}

# undef INIT_ARCH
# define INIT_ARCH() \
  void *vdso_gettimeofday = dl_vdso_vsym (HAVE_GETTIMEOFDAY_VSYSCALL)
libc_ifunc (__gettimeofday,
	    vdso_gettimeofday ? VDSO_IFUNC_RET (vdso_gettimeofday)
			      : (void *) __gettimeofday_syscall)

# else
int
__gettimeofday (struct timeval *restrict tv, void *restrict tz)
{
  if (__glibc_unlikely (tz != 0))
    memset (tz, 0, sizeof *tz);

  return INLINE_VSYSCALL (gettimeofday, 2, tv, tz);
}
# endif
weak_alias (__gettimeofday, gettimeofday)
#else /* USE_IFUNC_GETTIMEOFDAY  */
/* Conversion of gettimeofday function to support 64 bit time on archs
   with __WORDSIZE == 32 and __TIMESIZE == 32/64  */
#include <errno.h>

int
__gettimeofday64 (struct __timeval64 *restrict tv, void *restrict tz)
{
  if (__glibc_unlikely (tz != 0))
    memset (tz, 0, sizeof (struct timezone));

  struct __timespec64 ts64;
  if (__clock_gettime64 (CLOCK_REALTIME, &ts64))
	  return -1;

  *tv = timespec64_to_timeval64 (ts64);
  return 0;
}

# if __TIMESIZE != 64
libc_hidden_def (__gettimeofday64)

int
__gettimeofday (struct timeval *restrict tv, void *restrict tz)
{
  struct __timeval64 tv64;
  if (__gettimeofday64 (&tv64, tz))
	  return -1;

  if (! in_time_t_range (tv64.tv_sec))
    {
      __set_errno (EOVERFLOW);
      return -1;
    }

  *tv = valid_timeval64_to_timeval (tv64);
  return 0;
}
# endif
weak_alias (__gettimeofday, gettimeofday)
#endif
