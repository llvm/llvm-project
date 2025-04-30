/* Check if path supports 64-bit time interfaces.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <unistd.h>
#include <support/check.h>
#include <support/support.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sysdep.h>
#endif

#ifdef __linux__
static int
utimesat_call (const char *path, const struct __timespec64 tsp[2])
{
# ifndef __NR_utimensat_time64
#  define __NR_utimensat_time64 __NR_utimensat
# endif
  return syscall (__NR_utimensat_time64, AT_FDCWD, path, &tsp[0], 0);
}
#endif

bool
support_path_support_time64_value (const char *path, int64_t at, int64_t mt)
{
#ifdef __linux__
  /* Obtain the original timestamps to restore at the end.  */
  struct statx ostx;
  TEST_VERIFY_EXIT (statx (AT_FDCWD, path, 0, STATX_BASIC_STATS, &ostx) == 0);

  const struct __timespec64 tsp[] = { { at, 0 }, { mt, 0 } };

  /* Return is kernel does not support __NR_utimensat_time64.  */
  if (utimesat_call (path, tsp) == -1)
    return false;

  /* Verify if the last access and last modification time match the ones
     obtained with statx.  */
  struct statx stx;
  TEST_VERIFY_EXIT (statx (AT_FDCWD, path, 0, STATX_BASIC_STATS, &stx) == 0);

  bool support = stx.stx_atime.tv_sec == tsp[0].tv_sec
		 && stx.stx_mtime.tv_sec == tsp[1].tv_sec;

  /* Reset to original timestamps.  */
  const struct __timespec64 otsp[] =
  {
    { ostx.stx_atime.tv_sec, ostx.stx_atime.tv_nsec },
    { ostx.stx_mtime.tv_sec, ostx.stx_mtime.tv_nsec },
  };
  TEST_VERIFY_EXIT (utimesat_call (path, otsp) == 0);

  return support;
#else
  return true;
#endif
}
