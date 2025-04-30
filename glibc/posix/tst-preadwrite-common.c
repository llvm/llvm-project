/* Common definitions for pread and pwrite.
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
#include <error.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static void do_prepare (void);
#define PREPARE(argc, argv)	do_prepare ()
static int do_test (void);
#define TEST_FUNCTION		do_test ()

/* This defines the `main' function and some more.  */
#include <test-skeleton.c>

/* These are for the temporary file we generate.  */
static char *name;
static int fd;

static void
do_prepare (void)
{
  fd = create_temp_file ("tst-preadwrite.", &name);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot create temporary file");
}


static ssize_t
do_test_with_offset (off_t offset)
{
  char buf[1000];
  char res[1000];
  int i;
  ssize_t ret;

  memset (buf, '\0', sizeof (buf));
  memset (res, '\xff', sizeof (res));

  if (write (fd, buf, sizeof (buf)) != sizeof (buf))
    error (EXIT_FAILURE, errno, "during write");

  for (i = 100; i < 200; ++i)
    buf[i] = i;
  ret = pwrite (fd, buf + 100, 100, offset + 100);
  if (ret == -1)
    error (EXIT_FAILURE, errno, "during pwrite");

  for (i = 450; i < 600; ++i)
    buf[i] = i;
  ret = pwrite (fd, buf + 450, 150, offset + 450);
  if (ret == -1)
    error (EXIT_FAILURE, errno, "during pwrite");

  ret = pread (fd, res, sizeof (buf) - 50, offset + 50);
  if (ret == -1)
    error (EXIT_FAILURE, errno, "during pread");

  if (memcmp (buf + 50, res, ret) != 0)
    {
      printf ("error: read of pread != write of pwrite\n");
      return -1;
    }

  return ret;
}
