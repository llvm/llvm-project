/* pthread_getcpuclockid -- Get POSIX clockid_t for a pthread_t.  Linux version
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <pthreadP.h>
#include <sys/time.h>
#include <tls.h>
#include <kernel-posix-cpu-timers.h>
#include <shlib-compat.h>

int
__pthread_getcpuclockid (pthread_t threadid, clockid_t *clockid)
{
  struct pthread *pd = (struct pthread *) threadid;

  /* Make sure the descriptor is valid.  */
  if (INVALID_TD_P (pd))
    /* Not a valid thread handle.  */
    return ESRCH;

  /* The clockid_t value is a simple computation from the TID.  */

  const clockid_t tidclock = MAKE_THREAD_CPUCLOCK (pd->tid, CPUCLOCK_SCHED);

  *clockid = tidclock;
  return 0;
}
versioned_symbol (libc, __pthread_getcpuclockid, pthread_getcpuclockid,
                  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_getcpuclockid, pthread_getcpuclockid,
               GLIBC_2_2);
#endif
