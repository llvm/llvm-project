/* Get a clockid_t for the process CPU clock of a given process.  Generic.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#include <shlib-compat.h>

int
__clock_getcpuclockid (pid_t pid, clockid_t *clock_id)
{
  /* We don't allow any process ID but our own.  */
  if (pid != 0 && pid != getpid ())
    return EPERM;

#ifdef CLOCK_PROCESS_CPUTIME_ID
  /* Store the number.  */
  *clock_id = CLOCK_PROCESS_CPUTIME_ID;

  return 0;
#else
  /* We don't have a timer for that.  */
  return ENOENT;
#endif
}
versioned_symbol (libc, __clock_getcpuclockid, clock_getcpuclockid, GLIBC_2_17);
/* clock_getcpuclockid moved to libc in version 2.17;
   old binaries may expect the symbol version it had in librt.  */
#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
compat_symbol (libc, __clock_getcpuclockid, clock_getcpuclockid, GLIBC_2_2);
#endif
