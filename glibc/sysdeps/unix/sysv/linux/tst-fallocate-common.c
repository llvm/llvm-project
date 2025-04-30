/* Basic fallocate test (no specific flags is checked).
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

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <support/support.h>
#include <support/check.h>
#include <support/temp_file.h>

#define XSTR(s) STR(S)
#define STR(s)  #s

static char *temp_filename;
static int temp_fd;

static void
do_prepare (int argc, char **argv)
{
  temp_fd = create_temp_file ("tst-fallocate.", &temp_filename);
  if (temp_fd == -1)
    FAIL_EXIT1 ("cannot create temporary file: %m");
  if (!support_descriptor_supports_holes (temp_fd))
    FAIL_UNSUPPORTED ("File %s does not support holes", temp_filename);
}
#define PREPARE do_prepare

static int
do_test_with_offset (off_t offset)
{
  int ret;
  struct stat finfo;
#define BLK_SIZE 1024
  char bwrite[BLK_SIZE] = { 0xf0 };
  char bread[BLK_SIZE];

  /* It tries to fallocate 1024 bytes from 'offset' and then write 1024 bytes.
     After both operation rewind the file descriptor and read 1024 bytes
     and check if both buffer have the same contents.  */
  ret = fallocate (temp_fd, 0, offset, BLK_SIZE);
  if (ret == -1)
    {
      /* fallocate might not be fully supported by underlying filesystem (for
	 instance some NFS versions).   */
      if (errno == EOPNOTSUPP)
	FAIL_EXIT (77, "fallocate not supported");
      FAIL_EXIT1 ("fallocate failed");
    }

  ret = fstat (temp_fd, &finfo);
  if (ret == -1)
    FAIL_EXIT1 ("fstat failed");

  if (finfo.st_size < (offset + BLK_SIZE))
    FAIL_EXIT1 ("size of first fallocate less than expected (%llu)",
		(long long unsigned int)offset + BLK_SIZE);

  if (lseek (temp_fd, offset, SEEK_SET) == (off_t) -1)
    FAIL_EXIT1 ("fseek (0, SEEK_SET) failed");

  if (write (temp_fd, bwrite, BLK_SIZE) != BLK_SIZE)
    FAIL_EXIT1 ("fail trying to write " XSTR (BLK_SIZE) " bytes");

  if (lseek (temp_fd, offset, SEEK_SET) == (off_t) -1)
    FAIL_EXIT1 ("fseek (0, SEEK_SET) failed");

  if (read (temp_fd, bread, BLK_SIZE) != BLK_SIZE)
    FAIL_EXIT1 ("fail trying to read " XSTR (BLK_SIZE) " bytes");

  if (memcmp (bwrite, bread, BLK_SIZE) != 0)
    FAIL_EXIT1 ("buffer written different than buffer readed");

  return 0;
}

/* This function is defined by the individual tests.  */
static int do_test (void);

#include <support/test-driver.c>
