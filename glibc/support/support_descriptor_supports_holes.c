/* Test for file system hole support.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <stdbool.h>
#include <support.h>
#include <support/check.h>
#include <sys/stat.h>
#include <xunistd.h>

int
support_descriptor_supports_holes (int fd)
{
  enum
    {
      /* Write offset for the enlarged file.  This value is arbitrary
         and hopefully large enough to trigger the creation of holes.
         We cannot use the file system block size as a reference here
         because it is incorrect for network file systems.  */
      write_offset = 16 * 1024 * 1024,

      /* Our write may add this number of additional blocks (see
         block_limit below): writing at offset 16M can require two data block
         indirections, each of which can be as large as 8KB on ext2, thus 32
         512B sectors.  */
      block_headroom = 32,
    };

  struct stat64 st;
  xfstat (fd, &st);
  if (!S_ISREG (st.st_mode))
    FAIL_EXIT1 ("descriptor %d does not refer to a regular file", fd);
  if (st.st_size != 0)
    FAIL_EXIT1 ("descriptor %d does not refer to an empty file", fd);
  if (st.st_blocks > block_headroom)
    FAIL_EXIT1 ("descriptor %d refers to a pre-allocated file (%lld blocks)",
                fd, (long long int) st.st_blocks);

  /* Write a single byte at the start of the file to compute the block
     usage for a single byte.  */
  xlseek (fd, 0, SEEK_SET);
  char b = '@';
  xwrite (fd, &b, 1);
  /* Attempt to bypass delayed allocation.  */
  TEST_COMPARE (fsync (fd), 0);
  xfstat (fd, &st);

  /* This limit is arbitrary.  The file system needs to store
     somewhere that data exists at the write offset, and this may
     moderately increase the number of blocks used by the file, in
     proportion to the initial block count, but not in proportion to
     the write offset.  */
  unsigned long long int block_limit = 2 * st.st_blocks + block_headroom;

  /* Write a single byte at 16 megabytes.  */
  xlseek (fd, write_offset, SEEK_SET);
  xwrite (fd, &b, 1);
  /* Attempt to bypass delayed allocation.  */
  TEST_COMPARE (fsync (fd), 0);
  xfstat (fd, &st);
  bool supports_holes = st.st_blocks <= block_limit;

  /* Also check that extending the file does not fill up holes.  */
  xftruncate (fd, 2 * write_offset);
  /* Attempt to bypass delayed allocation.  */
  TEST_COMPARE (fsync (fd), 0);
  xfstat (fd, &st);
  supports_holes = supports_holes && st.st_blocks <= block_limit;

  /* Return to a zero-length file.  */
  xftruncate (fd, 0);
  xlseek (fd, 0, SEEK_SET);

  return supports_holes;
}
