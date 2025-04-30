/* clock_getcpuclockid -- Get a clockid_t for process CPU time.  Linux version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <time.h>
#include <unistd.h>
#include "kernel-posix-cpu-timers.h"
#include <sysdep.h>
#include <shlib-compat.h>

int
__clock_getcpuclockid (pid_t pid, clockid_t *clock_id)
{
  /* The clockid_t value is a simple computation from the PID.
     But we do a clock_getres call to validate it.  */

  const clockid_t pidclock = MAKE_PROCESS_CPUCLOCK (pid, CPUCLOCK_SCHED);

#ifndef __NR_clock_getres_time64
# define __NR_clock_getres_time64 __NR_clock_getres
#endif
  int r = INTERNAL_SYSCALL_CALL (clock_getres_time64, pidclock, NULL);

#ifndef __ASSUME_TIME64_SYSCALLS
  if (r != 0 && r == -ENOSYS)
    r = INTERNAL_SYSCALL_CALL (clock_getres, pidclock, NULL);
#endif

  if (r == 0)
    {
      *clock_id = pidclock;
      return 0;
    }
  if (r == -EINVAL)
    /* The clock_getres system call checked the PID for us.  */
    return ESRCH;
  return -r;
}

versioned_symbol (libc, __clock_getcpuclockid, clock_getcpuclockid, GLIBC_2_17);
/* clock_getcpuclockid moved to libc in version 2.17;
   old binaries may expect the symbol version it had in librt.  */
#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
strong_alias (__clock_getcpuclockid, __clock_getcpuclockid_2);
compat_symbol (libc, __clock_getcpuclockid_2, clock_getcpuclockid, GLIBC_2_2);
#endif
