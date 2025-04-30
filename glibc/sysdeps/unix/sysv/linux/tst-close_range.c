/* Test for the close_range system call.
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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include <array_length.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/descriptors.h>
#include <support/support.h>
#include <support/xsched.h>
#include <support/xunistd.h>

#define NFDS 100

static int
open_multiple_temp_files (void)
{
  /* Check if the temporary file descriptor has no no gaps.  */
  int lowfd = xopen ("/dev/null", O_RDONLY, 0600);
  for (int i = 1; i <= NFDS; i++)
    TEST_COMPARE (xopen ("/dev/null", O_RDONLY, 0600),
		  lowfd + i);
  return lowfd;
}

static void
close_range_test_max_upper_limit (void)
{
  struct support_descriptors *descrs = support_descriptors_list ();

  int lowfd = open_multiple_temp_files ();

  {
    int r = close_range (lowfd, ~0U, 0);
    if (r == -1 && errno == ENOSYS)
      FAIL_UNSUPPORTED ("close_range not supported");
    TEST_COMPARE (r, 0);
  }

  support_descriptors_check (descrs);
  support_descriptors_free (descrs);
}

static void
close_range_test_common (int lowfd, unsigned int flags)
{
  const int maximum_fd = lowfd + NFDS;
  const int half_fd = lowfd + NFDS / 2;
  const int gap_1 = maximum_fd - 8;

  /* Close half of the descriptors and check result.  */
  TEST_COMPARE (close_range (lowfd, half_fd, flags), 0);
  for (int i = lowfd; i <= half_fd; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }
  for (int i = half_fd + 1; i < maximum_fd; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  /* Create some gaps, close up to a threshold, and check result.  */
  xclose (lowfd + 57);
  xclose (lowfd + 78);
  xclose (lowfd + 81);
  xclose (lowfd + 82);
  xclose (lowfd + 84);
  xclose (lowfd + 90);

  TEST_COMPARE (close_range (half_fd + 1, gap_1, flags), 0);
  for (int i = half_fd + 1; i < gap_1; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }
  for (int i = gap_1 + 1; i < maximum_fd; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  /* Close the remaining but the last one.  */
  TEST_COMPARE (close_range (gap_1 + 1, maximum_fd - 1, flags), 0);
  for (int i = gap_1 + 1; i < maximum_fd - 1; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }
  TEST_VERIFY (fcntl (maximum_fd, F_GETFL) > -1);

  /* Close the last one.  */
  TEST_COMPARE (close_range (maximum_fd, maximum_fd, flags), 0);
  TEST_COMPARE (fcntl (maximum_fd, F_GETFL), -1);
  TEST_COMPARE (errno, EBADF);
}

/* Basic tests: check if the syscall close ranges with and without gaps.  */
static void
close_range_test (void)
{
  struct support_descriptors *descrs = support_descriptors_list ();

  /* Check if the temporary file descriptor has no no gaps.  */
  int lowfd = open_multiple_temp_files ();

  close_range_test_common (lowfd, 0);

  /* Double check by check the /proc.  */
  support_descriptors_check (descrs);
  support_descriptors_free (descrs);
}

_Noreturn static int
close_range_test_fn (void *arg)
{
  int lowfd = (int) ((uintptr_t) arg);
  close_range_test_common (lowfd, 0);
  exit (EXIT_SUCCESS);
}

/* Check if a clone_range on a subprocess created with CLONE_FILES close
   the shared file descriptor table entries in the parent.  */
static void
close_range_test_subprocess (void)
{
  struct support_descriptors *descrs = support_descriptors_list ();

  /* Check if the temporary file descriptor has no no gaps.  */
  int lowfd = open_multiple_temp_files ();

  struct support_stack stack = support_stack_alloc (4096);

  pid_t pid = xclone (close_range_test_fn, (void*) (uintptr_t) lowfd,
		      stack.stack, stack.size, CLONE_FILES | SIGCHLD);
  TEST_VERIFY_EXIT (pid > 0);
  int status;
  xwaitpid (pid, &status, 0);
  TEST_VERIFY (WIFEXITED (status));
  TEST_COMPARE (WEXITSTATUS (status), 0);

  support_stack_free (&stack);

  for (int i = lowfd; i < NFDS; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) < 0);

  support_descriptors_check (descrs);
  support_descriptors_free (descrs);
}


_Noreturn static int
close_range_unshare_test_fn (void *arg)
{
  int lowfd = (int) ((uintptr_t) arg);
  close_range_test_common (lowfd, CLOSE_RANGE_UNSHARE);
  exit (EXIT_SUCCESS);
}

/* Check if a close_range with CLOSE_RANGE_UNSHARE issued from a subprocess
   created with CLONE_FILES does not close the parent file descriptor list.  */
static void
close_range_unshare_test (void)
{
  struct support_descriptors *descrs1 = support_descriptors_list ();

  /* Check if the temporary file descriptor has no no gaps.  */
  int lowfd = open_multiple_temp_files ();

  struct support_descriptors *descrs2 = support_descriptors_list ();

  struct support_stack stack = support_stack_alloc (4096);

  pid_t pid = xclone (close_range_unshare_test_fn, (void*) (uintptr_t) lowfd,
		      stack.stack, stack.size, CLONE_FILES | SIGCHLD);
  TEST_VERIFY_EXIT (pid > 0);
  int status;
  xwaitpid (pid, &status, 0);
  TEST_VERIFY (WIFEXITED (status));
  TEST_COMPARE (WEXITSTATUS (status), 0);

  support_stack_free (&stack);

  for (int i = 0; i < NFDS; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  support_descriptors_check (descrs2);
  support_descriptors_free (descrs2);

  TEST_COMPARE (close_range (lowfd, lowfd + NFDS, 0), 0);

  support_descriptors_check (descrs1);
  support_descriptors_free (descrs1);
}

static bool
is_in_array (int *arr, size_t len, int fd)
{
  bool r = false;
  for (int i = 0; i < len; i++)
    if (arr[i] == fd)
      return true;
  return r;
}

static void
close_range_cloexec_test (void)
{
  /* Check if the temporary file descriptor has no no gaps.  */
  const int lowfd = open_multiple_temp_files ();

  const int maximum_fd = lowfd + NFDS;
  const int half_fd = lowfd + NFDS / 2;
  const int gap_1 = maximum_fd - 8;

  /* Close half of the descriptors and check result.  */
  int r = close_range (lowfd, half_fd, CLOSE_RANGE_CLOEXEC);
  if (r == -1 && errno == EINVAL)
    {
      printf ("%s: CLOSE_RANGE_CLOEXEC not supported\n", __func__);
      return;
    }
  for (int i = lowfd; i <= half_fd; i++)
    {
      int flags = fcntl (i, F_GETFD);
      TEST_VERIFY (flags > -1);
      TEST_COMPARE (flags & FD_CLOEXEC, FD_CLOEXEC);
    }
  for (int i = half_fd + 1; i < maximum_fd; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  /* Create some gaps, close up to a threshold, and check result.  */
  static int gap_close[] = { 57, 78, 81, 82, 84, 90 };
  for (int i = 0; i < array_length (gap_close); i++)
    xclose (gap_close[i]);

  TEST_COMPARE (close_range (half_fd + 1, gap_1, CLOSE_RANGE_CLOEXEC), 0);
  for (int i = half_fd + 1; i < gap_1; i++)
    {
      int flags = fcntl (i, F_GETFD);
      if (is_in_array (gap_close, array_length (gap_close), i))
        TEST_COMPARE (flags, -1);
      else
        {
          TEST_VERIFY (flags > -1);
          TEST_COMPARE (flags & FD_CLOEXEC, FD_CLOEXEC);
        }
    }
  for (int i = gap_1 + 1; i < maximum_fd; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  /* Close the remaining but the last one.  */
  TEST_COMPARE (close_range (gap_1 + 1, maximum_fd - 1, CLOSE_RANGE_CLOEXEC),
                0);
  for (int i = gap_1 + 1; i < maximum_fd - 1; i++)
    {
      int flags = fcntl (i, F_GETFD);
      TEST_VERIFY (flags > -1);
      TEST_COMPARE (flags & FD_CLOEXEC, FD_CLOEXEC);
    }
  TEST_VERIFY (fcntl (maximum_fd, F_GETFL) > -1);

  /* Close the last one.  */
  TEST_COMPARE (close_range (maximum_fd, maximum_fd, CLOSE_RANGE_CLOEXEC), 0);
  {
    int flags = fcntl (maximum_fd, F_GETFD);
    TEST_VERIFY (flags > -1);
    TEST_COMPARE (flags & FD_CLOEXEC, FD_CLOEXEC);
  }
}

static int
do_test (void)
{
  close_range_test_max_upper_limit ();
  close_range_test ();
  close_range_test_subprocess ();
  close_range_unshare_test ();
  close_range_cloexec_test ();

  return 0;
}

#include <support/test-driver.c>
