/* Basic posix_fadvise64 tests.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#define _FILE_OFFSET_BITS 64
#include "tst-posix_fadvise-common.c"

static int
do_test (void)
{
  int ret = do_test_common ();
  if (ret == 1)
    return 1;

  /* Test passing a negative length.  The compat fadvise64 might use
     off64_t for size argument passing, so using -1 for len without
     _FILE_OFFSET_BITS might not trigger the length issue.  */
  if (posix_fadvise (temp_fd, 0, -1, POSIX_FADV_NORMAL) != EINVAL)
    FAIL_EXIT1 ("posix_fadvise with negative length did not return EINVAL");

  /* Check with some offset values larger than 32-bits.  */
  off_t offset = UINT32_MAX + 2048LL;
  if (posix_fadvise (temp_fd, 0, offset, POSIX_FADV_NORMAL) != 0)
    FAIL_EXIT1 ("posix_fadvise failed (offset = 0, len = %zd) failed",
		(ssize_t)offset);

  if (posix_fadvise (temp_fd, offset, 0, POSIX_FADV_NORMAL) != 0)
    FAIL_EXIT1 ("posix_fadvise failed (offset = %zd, len = 0) failed",
		(ssize_t)offset);

  return 0;
}
