/* tzset tests with crafted time zone data.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.

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

#define _GNU_SOURCE 1

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <support/check.h>

static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

/* Returns the name of a large TZ file.  */
static char *
create_tz_file (off64_t size)
{
  char *path;
  int fd = create_temp_file ("tst-tzset-", &path);
  if (fd < 0)
    exit (1);
  if (!support_descriptor_supports_holes (fd))
    FAIL_UNSUPPORTED ("File %s does not support holes", path);

  // Reopen for large-file support.
  close (fd);
  fd = open64 (path, O_WRONLY);
  if (fd < 0)
    {
      printf ("open64 (%s) failed: %m\n", path);
      exit (1);
    }

  static const char data[] = {
    0x54, 0x5a, 0x69, 0x66, 0x32, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x58, 0x54, 0x47, 0x00, 0x00, 0x00,
    0x54, 0x5a, 0x69, 0x66, 0x32, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x04, 0xf8, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x58, 0x54, 0x47, 0x00, 0x00,
    0x00, 0x0a, 0x58, 0x54, 0x47, 0x30, 0x0a
  };
  ssize_t ret = write (fd, data, sizeof (data));
  if (ret < 0)
    {
      printf ("write failed: %m\n");
      exit (1);
    }
  if ((size_t) ret != sizeof (data))
    {
      printf ("Short write\n");
      exit (1);
    }
  if (lseek64 (fd, size, SEEK_CUR) < 0)
    {
      printf ("lseek failed: %m\n");
      close (fd);
      return NULL;
    }
  if (write (fd, "", 1) != 1)
    {
      printf ("Single-byte write failed\n");
      close (fd);
      return NULL;
    }
  if (close (fd) != 0)
    {
      printf ("close failed: %m\n");
      exit (1);
    }
  return path;
}

static void
test_tz_file (off64_t size)
{
  char *path = create_tz_file (size);
  if (setenv ("TZ", path, 1) < 0)
    {
      printf ("setenv failed: %m\n");
      exit (1);
    }
  tzset ();
  free (path);
}

static int
do_test (void)
{
  /* Limit the size of the process.  Otherwise, some of the tests will
     consume a lot of resources.  */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_AS, &limit) != 0)
      {
	printf ("getrlimit (RLIMIT_AS) failed: %m\n");
	return 1;
      }
    long target = 512 * 1024 * 1024;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur > target)
      {
	limit.rlim_cur = 512 * 1024 * 1024;
	if (setrlimit (RLIMIT_AS, &limit) != 0)
	  {
	    printf ("setrlimit (RLIMIT_AS) failed: %m\n");
	    return 1;
	  }
      }
  }

  int errors = 0;
  for (int i = 1; i <= 4; ++i)
    {
      char tz[16];
      snprintf (tz, sizeof (tz), "XT%d", i);
      if (setenv ("TZ", tz, 1) < 0)
	{
	  printf ("setenv failed: %m\n");
	  return 1;
	}
      tzset ();
      if (strcmp (tzname[0], tz) == 0)
	{
	  printf ("Unexpected success for %s\n", tz);
	  ++errors;
	}
    }

  /* Large TZ files.  */

  /* This will succeed on 64-bit architectures, and fail on 32-bit
     architectures.  It used to crash on 32-bit.  */
  test_tz_file (64 * 1024 * 1024);

  /* This will fail on 64-bit and 32-bit architectures.  It used to
     cause a test timeout on 64-bit and crash on 32-bit if the TZ file
     open succeeded for some reason (it does not use O_LARGEFILE in
     regular builds).  */
  test_tz_file (4LL * 1024 * 1024 * 1024 - 6);

  /* Large TZ variables.  */
  {
    size_t length = 64 * 1024 * 1024;
    char *value = malloc (length + 1);
    if (value == NULL)
      {
	puts ("malloc failed: %m");
	return 1;
      }
    value[length] = '\0';

    memset (value, ' ', length);
    value[0] = 'U';
    value[1] = 'T';
    value[2] = 'C';
    if (setenv ("TZ", value, 1) < 0)
      {
	printf ("setenv failed: %m\n");
	return 1;
      }
    tzset ();

    memset (value, '0', length);
    value[0] = '<';
    value[length - 1] = '>';
    if (setenv ("TZ", value, 1) < 0)
      {
	printf ("setenv failed: %m\n");
	return 1;
      }
    tzset ();
  }

  return errors > 0;
}
