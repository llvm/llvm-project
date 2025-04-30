/* Common posix_fadvise tests definitions.
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
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <support/support.h>
#include <support/check.h>
#include <support/temp_file.h>

static char *temp_filename;
static int temp_fd;
static char fifoname[] = "/tmp/tst-posix_fadvise-fifo-XXXXXX";
static int fifofd;

static void
do_prepare (int argc, char **argv)
{
  temp_fd = create_temp_file ("tst-posix_fadvise.", &temp_filename);
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

/* Effectivelly testing posix_fadvise is hard because side effects are not
   observed without checking either performance or any kernel specific
   supplied information.  Also, the syscall is meant to be an advisory,
   so the kernel is free to use this information in any way it deems fit,
   including ignoring it.

   This test check for some invalid returned operation to check argument
   passing and if implementation follows POSIX error definition.  */
static int
do_test_common (void)
{
  /* Add some data to file and ensure it is written to disk.  */
#define BLK_SIZE 2048
  char buffer[BLK_SIZE] = { 0xcd };
  ssize_t ret;

  if ((ret = write (temp_fd, buffer, BLK_SIZE)) != BLK_SIZE)
    FAIL_EXIT1 ("write returned %zd different than expected %d",
		ret, BLK_SIZE);

  if (fsync (temp_fd) != 0)
    FAIL_EXIT1 ("fsync failed");

  /* Test passing an invalid fd.  */
  if (posix_fadvise (-1, 0, 0, POSIX_FADV_NORMAL) != EBADF)
    FAIL_EXIT1 ("posix_fadvise with invalid fd did not return EBADF");

  /* Test passing an invalid operation.  */
  if (posix_fadvise (temp_fd, 0, 0, -1) != EINVAL)
    FAIL_EXIT1 ("posix_fadvise with invalid advise did not return EINVAL");

  /* Test passing a FIFO fd.  */
  if (posix_fadvise (fifofd, 0, 0, POSIX_FADV_NORMAL) != ESPIPE)
    FAIL_EXIT1 ("posix_advise with PIPE fd did not return ESPIPE");

  /* Default fadvise on all file starting at initial position.  */
  if (posix_fadvise (temp_fd, 0, 0, POSIX_FADV_NORMAL) != 0)
    FAIL_EXIT1 ("default posix_fadvise failed");

  if (posix_fadvise (temp_fd, 0, 2 * BLK_SIZE, POSIX_FADV_NORMAL) != 0)
    FAIL_EXIT1 ("posix_fadvise failed (offset = 0, len = %d) failed",
		BLK_SIZE);

  if (posix_fadvise (temp_fd, 2 * BLK_SIZE, 0, POSIX_FADV_NORMAL) != 0)
    FAIL_EXIT1 ("posix_fadvise failed (offset = %d, len = 0) failed",
		BLK_SIZE);

  return 0;
}

#define PREPARE do_prepare

/* This function is defined by the individual tests.  */
static int do_test (void);

#include <support/test-driver.c>
