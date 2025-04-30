/* Basic sync_file_range (not specific flag is checked).
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

/* sync_file_range is only define for LFS.  */
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <support/temp_file.h>
#include <support/check.h>

#define XSTR(s) STR(S)
#define STR(s)  #s

static char *temp_filename;
static int temp_fd;

static char fifoname[] = "/tmp/tst-posix_fadvise-fifo-XXXXXX";
static int fifofd;

void
do_prepare (int argc, char **argv)
{
  temp_fd = create_temp_file ("tst-file_sync_range.", &temp_filename);
  if (temp_fd == -1)
    FAIL_EXIT1 ("cannot create temporary file: %m");

  if (mktemp (fifoname) == NULL)
    FAIL_EXIT1 ("cannot generate temp file name: %m");
  add_temp_file (fifoname);

  if (mkfifo (fifoname, S_IWUSR | S_IRUSR) != 0)
    FAIL_EXIT1 ("cannot create fifo: %m");

  fifofd = open (fifoname, O_RDONLY | O_NONBLOCK);
  if (fifofd == -1)
    FAIL_EXIT1 ("cannot open fifo: %m");
}
#define PREPARE do_prepare

static int
do_test (void)
{
  int ret;

  /* This tests first check for some invalid usage and then check for
     a simple usage.  It does not cover for all possible issue since for
     EIO/ENOMEM/ENOSPC would require to create very specific scenarios that
     are outside the current test coverage (basically correct kernel argument
     passing.  */

  /* Check for invalid file descriptor.  */
  if ((ret = sync_file_range (-1, 0, 0, 0)) != -1)
    FAIL_EXIT1 ("sync_file_range did not fail on an invalid descriptor "
		"(returned %d, expected -1)", ret);
  if (errno != EBADF)
    FAIL_EXIT1 ("sync_file_range on an invalid descriptor did not set errno to "
		"EBADF (%d)", errno);

  if ((ret = sync_file_range (fifofd, 0, 0, 0)) != -1)
    FAIL_EXIT1 ("sync_file_range did not fail on an invalid descriptor "
		"(returned %d, expected -1)", ret);
  if (errno != ESPIPE)
    FAIL_EXIT1 ("sync_file_range on an invalid descriptor did not set errno to "
		"EBADF (%d)", errno);

  /* Check for invalid flags (it must be
     SYNC_FILE_RANGE_{WAIT_BEFORE,WRITE,WAIT_AFTER) or a 'or' combination of
     them.  */
  if ((ret = sync_file_range (temp_fd, 0, 0, -1)) != -1)
    FAIL_EXIT1 ("sync_file_range did not failed with invalid flags "
		"(returned %d, " "expected -1)", ret);
  if (errno != EINVAL)
    FAIL_EXIT1 ("sync_file_range with invalid flag did not set errno to "
		"EINVAL (%d)", errno);

  /* Check for negative offset.  */
  if ((ret = sync_file_range (temp_fd, -1, 1, 0)) != -1)
    FAIL_EXIT1 ("sync_file_range did not failed with invalid offset "
		"(returned %d, expected -1)", ret);
  if (errno != EINVAL)
    FAIL_EXIT1 ("sync_file_range with invalid offset did not set errno to "
		"EINVAL (%d)", errno);

  /* offset + nbytes must be a positive value.  */
  if ((ret = sync_file_range (temp_fd, 1024, -2048, 0)) != -1)
    FAIL_EXIT1 ("sync_file_range did not failed with invalid nbytes (returned %d, "
	  "expected -1)", ret);
  if (errno != EINVAL)
    FAIL_EXIT1 ("sync_file_range with invalid offset did not set errno to "
		"EINVAL (%d)", errno);

  /* offset + nbytes must be larger or equal than offset */
  if ((ret = sync_file_range (temp_fd, -1024, 1024, 0)) != -1)
    FAIL_EXIT1 ("sync_file_range did not failed with invalid offset "
		"(returned %d, expected -1)", ret);
  if (errno != EINVAL)
    FAIL_EXIT1 ("sync_file_range with invalid offset did not set errno to "
		"EINVAL (%d)", errno);

  /* Check simple successful case.  */
  if ((ret = sync_file_range (temp_fd, 0, 1024, 0)) == -1)
    FAIL_EXIT1 ("sync_file_range failed (errno = %d)", errno);

  /* Finally check also a successful case with a 64-bit offset.  */
  off_t large_offset = UINT32_MAX + 2048LL;
  if ((ret = sync_file_range (temp_fd, large_offset, 1024, 0)) == -1)
    FAIL_EXIT1 ("sync_file_range failed (errno = %d)", errno);

  return 0;
}

#include <support/test-driver.c>
