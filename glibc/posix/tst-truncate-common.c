/* Common f{truncate} tests definitions.
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

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

static void do_prepare (void);
#define PREPARE(argc, argv)     do_prepare ()
static int do_test (void);
#define TEST_FUNCTION           do_test ()

#include <test-skeleton.c>

static char *temp_filename;
static int temp_fd;

static void
do_prepare (void)
{
  temp_fd = create_temp_file ("tst-trucate.", &temp_filename);
  if (temp_fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }
}

#define FAIL(str) \
  do { printf ("error: %s (line %d)\n", str, __LINE__); return 1; } while (0)

static int
do_test_with_offset (off_t offset)
{
  struct stat st;
  char buf[1000];

  memset (buf, 0xcf, sizeof (buf));

  if (pwrite (temp_fd, buf, sizeof (buf), offset) != sizeof (buf))
    FAIL ("write failed");
  if (fstat (temp_fd, &st) < 0 || st.st_size != (offset + sizeof (buf)))
    FAIL ("initial size wrong");

  if (ftruncate (temp_fd, offset + 800) < 0)
    FAIL ("size reduction with ftruncate failed");
  if (fstat (temp_fd, &st) < 0 || st.st_size != (offset + 800))
    FAIL ("size after reduction with ftruncate is incorrect");

  /* The following test covers more than POSIX.  POSIX does not require
     that ftruncate() can increase the file size.  But we are testing
     Unix systems.  */
  if (ftruncate (temp_fd, offset + 1200) < 0)
    FAIL ("size increate with ftruncate failed");
  if (fstat (temp_fd, &st) < 0 || st.st_size != (offset + 1200))
    FAIL ("size after increase is incorrect");

  if (truncate (temp_filename, offset + 800) < 0)
    FAIL ("size reduction with truncate failed");
  if (fstat (temp_fd, &st) < 0 || st.st_size != (offset + 800))
    FAIL ("size after reduction with truncate incorrect");

  /* The following test covers more than POSIX.  POSIX does not require
     that truncate() can increase the file size.  But we are testing
     Unix systems.  */
  if (truncate (temp_filename, (offset + 1200)) < 0)
    FAIL ("size increase with truncate failed");
  if (fstat (temp_fd, &st) < 0 || st.st_size != (offset + 1200))
    FAIL ("size increase with truncate is incorrect");

  return 0;
}
