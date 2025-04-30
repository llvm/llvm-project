/* Test POSIX lock on an open file (lockf).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.
*/

#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <stdio.h>

#include <support/temp_file.h>
#include <support/capture_subprocess.h>
#include <support/check.h>

static char *temp_filename;
static int temp_fd;

static void
do_prepare (int argc, char **argv)
{
  temp_fd = create_temp_file ("tst-lockfd.", &temp_filename);
  TEST_VERIFY_EXIT (temp_fd != -1);
}
#define PREPARE do_prepare

static void
do_test_child_lockf (void *closure)
{
  /* Check if parent has [0, 1024) locked.  */
  TEST_COMPARE (lseek (temp_fd, 0, SEEK_SET), 0);
  TEST_COMPARE (lockf (temp_fd, F_TLOCK, 1024), -1);
  TEST_COMPARE (errno, EAGAIN);
  TEST_COMPARE (lockf (temp_fd, F_TEST, 1024), -1);
  TEST_COMPARE (errno, EACCES);
  /* Also Check if parent has last 1024 bytes locked.  */
  TEST_COMPARE (lseek (temp_fd, INT32_MAX-1024, SEEK_SET), INT32_MAX-1024);
  TEST_COMPARE (lockf (temp_fd, F_TEST, 1024), -1);

  /* And try to lock [1024, 2048).  */
  TEST_COMPARE (lseek (temp_fd, 1024, SEEK_SET), 1024);
  TEST_COMPARE (lockf (temp_fd, F_LOCK, 1024), 0);

  /* Check if non-LFS interface cap access to 32-bif off_t.  */
  TEST_COMPARE (lseek64 (temp_fd, (off64_t)INT32_MAX, SEEK_SET),
		(off64_t)INT32_MAX);
  TEST_COMPARE (lockf64 (temp_fd, F_TEST, 1024), 0);
}

static void
do_test_child_lockf64 (void *closure)
{
  /* Check if parent has [0, 1024) locked.  */
  TEST_COMPARE (lseek64 (temp_fd, 0, SEEK_SET), 0);
  TEST_COMPARE (lockf64 (temp_fd, F_TLOCK, 1024), -1);
  TEST_COMPARE (errno, EAGAIN);
  TEST_COMPARE (lockf64 (temp_fd, F_TEST, 1024), -1);
  TEST_COMPARE (errno, EACCES);
  /* Also Check if parent has last 1024 bytes locked.  */
  TEST_COMPARE (lseek64 (temp_fd, INT32_MAX-1024, SEEK_SET), INT32_MAX-1024);
  TEST_COMPARE (lockf64 (temp_fd, F_TEST, 1024), -1);

  /* And try to lock [1024, 2048).  */
  TEST_COMPARE (lseek64 (temp_fd, 1024, SEEK_SET), 1024);
  TEST_COMPARE (lockf64 (temp_fd, F_LOCK, 1024), 0);

  /* And also [INT32_MAX, INT32_MAX+1024).  */
  {
    off64_t off = (off64_t)INT32_MAX;
    TEST_COMPARE (lseek64 (temp_fd, off, SEEK_SET), off);
    TEST_COMPARE (lockf64 (temp_fd, F_LOCK, 1024), 0);
  }

  /* Check if [INT32_MAX+1024, INT64_MAX) is locked.  */
  {
    off64_t off = (off64_t)INT32_MAX+1024;
    TEST_COMPARE (lseek64 (temp_fd, off, SEEK_SET), off);
    TEST_COMPARE (lockf64 (temp_fd, F_TLOCK, 1024), -1);
    TEST_COMPARE (errno, EAGAIN);
    TEST_COMPARE (lockf64 (temp_fd, F_TEST, 1024), -1);
    TEST_COMPARE (errno, EACCES);
  }
}

static int
do_test (void)
{
  /* Basic tests to check if a lock can be obtained and checked.  */
  TEST_COMPARE (lockf (temp_fd, F_LOCK, 1024), 0);
  TEST_COMPARE (lockf (temp_fd, F_LOCK, INT32_MAX), 0);
  TEST_COMPARE (lockf (temp_fd, F_TLOCK, 1024), 0);
  TEST_COMPARE (lockf (temp_fd, F_TEST, 1024), 0);
  TEST_COMPARE (lseek (temp_fd, 1024, SEEK_SET), 1024);
  TEST_COMPARE (lockf (temp_fd, F_ULOCK, 1024), 0);
  /* Parent process should have ([0, 1024), [2048, INT32_MAX)) ranges locked.  */

  {
    struct support_capture_subprocess result;
    result = support_capture_subprocess (do_test_child_lockf, NULL);
    support_capture_subprocess_check (&result, "lockf", 0, sc_allow_none);
  }

  if (sizeof (off_t) != sizeof (off64_t))
    {
      /* Check if previously locked regions with LFS symbol.  */
      TEST_COMPARE (lseek (temp_fd, 0, SEEK_SET), 0);
      TEST_COMPARE (lockf64 (temp_fd, F_LOCK, 1024), 0);
      TEST_COMPARE (lockf64 (temp_fd, F_TLOCK, 1024), 0);
      TEST_COMPARE (lockf64 (temp_fd, F_TEST, 1024), 0);
      /* Lock region [INT32_MAX+1024, INT64_MAX).  */
      off64_t off = (off64_t)INT32_MAX + 1024;
      TEST_COMPARE (lseek64 (temp_fd, off, SEEK_SET), off);
      TEST_COMPARE (lockf64 (temp_fd, F_LOCK, 1024), 0);
      /* Parent process should have ([0, 1024), [2048, INT32_MAX),
	 [INT32_MAX+1024, INT64_MAX)) ranges locked.  */

      {
	struct support_capture_subprocess result;
	result = support_capture_subprocess (do_test_child_lockf64, NULL);
	support_capture_subprocess_check (&result, "lockf", 0, sc_allow_none);
      }
    }

  return 0;
}

#include <support/test-driver.c>
