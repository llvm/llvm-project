/* Tests for pread64 and pwrite64.
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
#include "tst-preadvwritev-common.c"

static int
do_test (void)
{
  int ret;

  ret = do_test_with_offset (0);

  if (!temp_fd_supports_holes)
    {
      puts ("warning: partial test due to lack of support for holes");
      return ret;
    }

  /* Create a sparse file larger than 4GB to check if offset is handled
     correctly in p{write,read}v64. */
  off_t base_offset = UINT32_MAX + 2048LL;
  ret += do_test_with_offset (base_offset);

  struct stat st;
  if (fstat (temp_fd, &st) == -1)
    {
      printf ("error: fstat on temporary file failed: %m");
      return 1;
    }

  /* The total size should base_offset plus 2 * 96.  */
  off_t expected_value = base_offset + (2 * (96LL));
  if (st.st_size != expected_value)
    {
      printf ("error: file size different than expected (%jd != %jd)\n",
	      (intmax_t) expected_value, (intmax_t) st.st_size);
      return 1;
    }

  return ret;
}
