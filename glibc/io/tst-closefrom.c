/* Smoke test for the closefrom.
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

#include <errno.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <support/check.h>
#include <support/descriptors.h>
#include <support/xunistd.h>

#include <array_length.h>

#define NFDS 100

static int
open_multiple_temp_files (void)
{
  /* Check if the temporary file descriptor has no no gaps.  */
  int lowfd = xopen ("/dev/null", O_RDONLY, 0600);
  for (int i = 1; i <= NFDS; i++)
    TEST_COMPARE (xopen ("/dev/null", O_RDONLY, 0600), lowfd + i);
  return lowfd;
}

static int
closefrom_test (void)
{
  struct support_descriptors *descrs = support_descriptors_list ();

  int lowfd = open_multiple_temp_files ();

  const int maximum_fd = lowfd + NFDS;
  const int half_fd = lowfd + NFDS / 2;
  const int gap = maximum_fd / 4;

  /* Close half of the descriptors and check result.  */
  closefrom (half_fd);

  for (int i = half_fd; i <= maximum_fd; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }
  for (int i = 0; i < half_fd; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  /* Create some gaps, close up to a threshold, and check result.  */
  xclose (lowfd + 35);
  xclose (lowfd + 38);
  xclose (lowfd + 42);
  xclose (lowfd + 46);

  /* Close half of the descriptors and check result.  */
  closefrom (gap);
  for (int i = gap + 1; i < maximum_fd; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }
  for (int i = 0; i < gap; i++)
    TEST_VERIFY (fcntl (i, F_GETFL) > -1);

  /* Close the remmaining but the last one.  */
  closefrom (lowfd + 1);
  for (int i = lowfd + 1; i <= maximum_fd; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }
  TEST_VERIFY (fcntl (lowfd, F_GETFL) > -1);

  /* Close the last one.  */
  closefrom (lowfd);
  TEST_COMPARE (fcntl (lowfd, F_GETFL), -1);
  TEST_COMPARE (errno, EBADF);

  /* Double check by check the /proc.  */
  support_descriptors_check (descrs);
  support_descriptors_free (descrs);

  return 0;
}

/* Check if closefrom works even when no new file descriptors can be
   created.  */
static int
closefrom_test_file_desc_limit (void)
{
  int max_fd = NFDS;
  {
    struct rlimit rl;
    if (getrlimit (RLIMIT_NOFILE, &rl) == -1)
      FAIL_EXIT1 ("getrlimit (RLIMIT_NOFILE): %m");

    max_fd = (rl.rlim_cur < max_fd ? rl.rlim_cur : max_fd);
    rl.rlim_cur = max_fd;

    if (setrlimit (RLIMIT_NOFILE, &rl) == 1)
      FAIL_EXIT1 ("setrlimit (RLIMIT_NOFILE): %m");
  }

  /* Exhauste the file descriptor limit.  */
  int lowfd = xopen ("/dev/null", O_RDONLY, 0600);
  for (;;)
    {
      int fd = open ("/dev/null", O_RDONLY, 0600);
      if (fd == -1)
	{
	  if (errno != EMFILE)
	    FAIL_EXIT1 ("open: %m");
	  break;
	}
      TEST_VERIFY_EXIT (fd < max_fd);
    }

  closefrom (lowfd);
  for (int i = lowfd; i < NFDS; i++)
    {
      TEST_COMPARE (fcntl (i, F_GETFL), -1);
      TEST_COMPARE (errno, EBADF);
    }

  return 0;
}

static int
do_test (void)
{
  closefrom_test ();
  closefrom_test_file_desc_limit ();

  return 0;
}

#include <support/test-driver.c>
