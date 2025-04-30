/* Test for /proc/self/fd (or /dev/fd) pathname construction.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <fd_to_filename.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xunistd.h>

/* Run a check on one value.  */
static void
check (int value)
{
  if (value < 0)
    /* Negative descriptor values violate the precondition.  */
    return;

  struct fd_to_filename storage;
  char *actual = __fd_to_filename (value, &storage);
  char expected[100];
  snprintf (expected, sizeof (expected), FD_TO_FILENAME_PREFIX "%d", value);
  TEST_COMPARE_STRING (actual, expected);
}

/* Check various ranges constructed around powers.  */
static void
check_ranges (int base)
{
  unsigned int power = 1;
  do
    {
      for (int factor = 1; factor < base; ++factor)
        for (int shift = -1000; shift <= 1000; ++shift)
          check (factor * power + shift);
    }
  while (!__builtin_mul_overflow (power, base, &power));
}

/* Check that it is actually possible to use a the constructed
   name.  */
static void
check_open (void)
{
  int pipes[2];
  xpipe (pipes);

  struct fd_to_filename storage;
  int read_alias = xopen (__fd_to_filename (pipes[0], &storage), O_RDONLY, 0);
  int write_alias = xopen (__fd_to_filename (pipes[1], &storage), O_WRONLY, 0);

  /* Ensure that all the descriptor numbers are different.  */
  TEST_VERIFY (pipes[0] < pipes[1]);
  TEST_VERIFY (pipes[1] < read_alias);
  TEST_VERIFY (read_alias < write_alias);

  xwrite (write_alias, "1", 1);
  char buf[16];
  TEST_COMPARE_BLOB ("1", 1, buf, read (pipes[0], buf, sizeof (buf)));

  xwrite (pipes[1], "2", 1);
  TEST_COMPARE_BLOB ("2", 1, buf, read (read_alias, buf, sizeof (buf)));

  xwrite (write_alias, "3", 1);
  TEST_COMPARE_BLOB ("3", 1, buf, read (read_alias, buf, sizeof (buf)));

  xwrite (pipes[1], "4", 1);
  TEST_COMPARE_BLOB ("4", 1, buf, read (pipes[0], buf, sizeof (buf)));

  xclose (write_alias);
  xclose (read_alias);
  xclose (pipes[1]);
  xclose (pipes[0]);
}

static int
do_test (void)
{
  check_ranges (2);
  check_ranges (10);

  check_open ();

  return 0;
}

#include <support/test-driver.c>
