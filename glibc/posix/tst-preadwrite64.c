/* Tests for pread64 and pwrite64.
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

#define _FILE_OFFSET_BITS 64
#include "tst-preadwrite-common.c"

static int
do_test (void)
{
  ssize_t ret;

  ret = do_test_with_offset (0);
  if (ret == -1)
    return 1;

  /* Create a sparse file larger than 4GB to check if offset is handled
     correctly in p{write,read}64. */
  off_t base_offset = UINT32_MAX + 2048LL;
  ret = do_test_with_offset (base_offset);
  if (ret == -1)
    return 1;

  struct stat st;
  if (fstat (fd, &st) == -1)
    {
      printf ("error: fstat on temporary file failed: %m");
      return 1;
    }

  /* The file size should >= base_offset plus bytes read.  */
  off_t expected_value = base_offset + ret;
  if (st.st_size < expected_value)
    {
      printf ("error: file size less than expected (%jd > %jd)\n",
	      (intmax_t) expected_value, (intmax_t) st.st_size);
      return 1;
    }

  return 0;
}
