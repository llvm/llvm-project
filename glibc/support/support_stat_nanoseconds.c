/* Check if stat supports nanosecond resolution.
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

#include <errno.h>
#include <fcntl.h>
#include <support/check.h>
#include <support/support.h>
#include <support/timespec.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

bool
support_stat_nanoseconds (const char *path)
{
  bool support = true;
#ifdef __linux__
  /* Obtain the original timestamp to restore at the end.  */
  struct stat ost;
  TEST_VERIFY_EXIT (stat (path, &ost) == 0);

  const struct timespec tsp[] = { { 0, TIMESPEC_HZ - 1 },
				  { 0, TIMESPEC_HZ / 2 } };
  TEST_VERIFY_EXIT (utimensat (AT_FDCWD, path, tsp, 0) == 0);

  struct stat st;
  TEST_VERIFY_EXIT (stat (path, &st) == 0);

  support = st.st_atim.tv_nsec == tsp[0].tv_nsec
	    && st.st_mtim.tv_nsec == tsp[1].tv_nsec;

  /* Reset to original timestamps.  */
  const struct timespec otsp[] =
  {
    { ost.st_atim.tv_sec, ost.st_atim.tv_nsec },
    { ost.st_mtim.tv_sec, ost.st_mtim.tv_nsec },
  };
  TEST_VERIFY_EXIT (utimensat (AT_FDCWD, path, otsp, 0) == 0);
#endif
  return support;
}
