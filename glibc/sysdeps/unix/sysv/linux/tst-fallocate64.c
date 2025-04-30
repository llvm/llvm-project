/* Basic fallocate64 test (no specific flags is checked).
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
#include "tst-fallocate-common.c"

static int
do_test (void)
{
  ssize_t ret;

  ret = do_test_with_offset (0);
  if (ret == 1)
    return 1;

  off_t base_offset = UINT32_MAX + 2048LL;
  ret = do_test_with_offset (base_offset);
  if (ret == 1)
    return 1;

  struct stat st;
  if (fstat (temp_fd, &st) == -1)
    FAIL_EXIT1 ("fstat on temporary file failed: %m");

  /* The file size should >= base_offset plus bytes written.  */
  off_t expected_value = base_offset + ret;
  if (st.st_size < expected_value)
    FAIL_EXIT1 ("file size less than expected (%jd > %jd)",
		(intmax_t) expected_value, (intmax_t) st.st_size);

  return 0;
}
