/* time -- Get number of seconds since Epoch.  Linux version.
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
   <https://www.gnu.org/licenses/>.  */

/* Optimize the function call by setting the PLT directly to vDSO symbol.  */
#ifdef USE_IFUNC_TIME
# include <time.h>
# include <sysdep.h>
# include <sysdep-vdso.h>

#ifdef SHARED
# include <dl-vdso.h>
# include <libc-vdso.h>

static time_t
time_syscall (time_t *t)
{
  return INLINE_SYSCALL_CALL (time, t);
}

# undef INIT_ARCH
# define INIT_ARCH() \
  void *vdso_time = dl_vdso_vsym (HAVE_TIME_VSYSCALL);
libc_ifunc (time,
	    vdso_time ? VDSO_IFUNC_RET (vdso_time)
		      : (void *) time_syscall);

# else
time_t
time (time_t *t)
{
  return INLINE_VSYSCALL (time, 1, t);
}
# endif /* !SHARED */
#else /* USE_IFUNC_TIME  */
# include <time.h>
# include <time-clockid.h>
# include <errno.h>

/* Return the time now, and store it in *TIMER if not NULL.  */

__time64_t
__time64 (__time64_t *timer)
{
  struct __timespec64 ts;
  __clock_gettime64 (TIME_CLOCK_GETTIME_CLOCKID, &ts);

  if (timer != NULL)
    *timer = ts.tv_sec;
  return ts.tv_sec;
}

# if __TIMESIZE != 64
libc_hidden_def (__time64)

time_t
__time (time_t *timer)
{
  __time64_t t = __time64 (NULL);

  if (! in_time_t_range (t))
    {
      __set_errno (EOVERFLOW);
      return -1;
    }

  if (timer != NULL)
    *timer = t;
  return t;
}
# endif
weak_alias (__time, time)
#endif
