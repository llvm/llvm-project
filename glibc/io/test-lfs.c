/* Some basic tests for LFS.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2000.

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
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <error.h>
#include <errno.h>
#include <sys/resource.h>
#include <support/check.h>

/* Prototype for our test function.  */
extern void do_prepare (int argc, char *argv[]);
extern int do_test (int argc, char *argv[]);

/* We have a preparation function.  */
#define PREPARE do_prepare

/* This defines the `main' function and some more.  */
#include <test-skeleton.c>

/* These are for the temporary file we generate.  */
char *name;
int fd;

/* 2^31 = 2GB.  */
#define TWO_GB 2147483648LL

void
do_prepare (int argc, char *argv[])
{
  size_t name_len;
  struct rlimit64 rlim;

  name_len = strlen (test_dir);
  name = xmalloc (name_len + sizeof ("/lfsXXXXXX"));
  mempcpy (mempcpy (name, test_dir, name_len),
           "/lfsXXXXXX", sizeof ("/lfsXXXXXX"));

  /* Open our test file.   */
  fd = mkstemp64 (name);
  if (fd == -1)
    {
      if (errno == ENOSYS)
	{
	  /* Fail silently.  */
	  error (0, 0, "open64 is not supported");
	  exit (EXIT_SUCCESS);
	}
      else
	error (EXIT_FAILURE, errno, "cannot create temporary file");
    }
  if (!support_descriptor_supports_holes (fd))
    FAIL_UNSUPPORTED ("File %s does not support holes", name);
  add_temp_file (name);

  if (getrlimit64 (RLIMIT_FSIZE, &rlim) != 0)
    {
      error (0, errno, "cannot get resource limit");
      exit (0);
    }
  if (rlim.rlim_cur < TWO_GB + 200)
    {
      rlim.rlim_cur = TWO_GB + 200;
      if (setrlimit64 (RLIMIT_FSIZE, &rlim) != 0)
	{
	  error (0, errno, "cannot reset file size limits");
	  exit (0);
	}
    }
}

static void
test_ftello (void)
{
  FILE *f;
  int ret;
  off64_t pos;

  f = fopen64 (name, "w");

  ret = fseeko64 (f, TWO_GB+100, SEEK_SET);
  if (ret == -1 && errno == ENOSYS)
    {
      error (0, 0, "fseeko64 is not supported.");
      exit (EXIT_SUCCESS);
    }
  if (ret == -1 && errno == EINVAL)
    {
      error (0, 0, "LFS seems not to be supported");
      exit (EXIT_SUCCESS);
    }
  if (ret == -1)
    {
      error (0, errno, "fseeko64 failed with error");
      exit (EXIT_FAILURE);
    }

  ret = fwrite ("Hello", 1, 5, f);
  if (ret == -1 && errno == EFBIG)
    {
      error (0, errno, "LFS seems not to be supported");
      exit (EXIT_SUCCESS);
    }

  if (ret == -1 && errno == ENOSPC)
    {
      error (0, 0, "Not enough space to write file.");
      exit (EXIT_SUCCESS);
    }

  if (ret != 5)
    error (EXIT_FAILURE, errno, "Cannot write test string to large file");

  pos = ftello64 (f);

  if (pos != TWO_GB+105)
    {
      error (0, 0, "ftello64 gives wrong result.");
      exit (EXIT_FAILURE);
    }

  fclose (f);
}

int
do_test (int argc, char *argv[])
{
  int ret, fd2;
  struct stat64 statbuf;

  ret = lseek64 (fd, TWO_GB+100, SEEK_SET);
  if (ret == -1 && errno == ENOSYS)
    {
      error (0, 0, "lseek64 is not supported.");
      exit (EXIT_SUCCESS);
    }
  if (ret == -1 && errno == EINVAL)
    {
      error (0, 0, "LFS seems not to be supported.");
      exit (EXIT_SUCCESS);
    }
  if (ret == -1)
    {
      error (0, errno, "lseek64 failed with error");
      exit (EXIT_FAILURE);
    }
  off64_t offset64 = lseek64 (fd, 0, SEEK_CUR);
  if (offset64 != TWO_GB + 100)
    {
      error (0, 0, "lseek64 did not return expected offset");
      exit (EXIT_FAILURE);
    }
  off_t offset = lseek (fd, 0, SEEK_CUR);
  if (sizeof (off_t) < sizeof (off64_t))
    {
      if (offset != -1 || errno != EOVERFLOW)
	{
	  error (0, 0, "lseek did not fail with EOVERFLOW");
	  exit (EXIT_FAILURE);
	}
    }
  else
    if (offset != TWO_GB + 100)
      {
	error (0, 0, "lseek did not return expected offset");
	exit (EXIT_FAILURE);
      }

  ret = write (fd, "Hello", 5);
  if (ret == -1 && errno == EFBIG)
    {
      error (0, 0, "LFS seems not to be supported.");
      exit (EXIT_SUCCESS);
    }

  if (ret == -1 && errno == ENOSPC)
    {
      error (0, 0, "Not enough space to write file.");
      exit (EXIT_SUCCESS);
    }

  if (ret != 5)
    error (EXIT_FAILURE, errno, "cannot write test string to large file");

  ret = close (fd);

  if (ret == -1)
    error (EXIT_FAILURE, errno, "error closing file");

  ret = stat64 (name, &statbuf);

  if (ret == -1 && (errno == ENOSYS || errno == EOVERFLOW))
    error (0, 0, "stat64 is not supported.");
  else if (ret == -1)
    error (EXIT_FAILURE, errno, "cannot stat file `%s'", name);
  else if (statbuf.st_size != (TWO_GB + 100 + 5))
    error (EXIT_FAILURE, 0, "stat reported size %lld instead of %lld.",
	   (long long int) statbuf.st_size, (TWO_GB + 100 + 5));

  fd2 = openat64 (AT_FDCWD, name, O_RDWR);
  if (fd2 == -1)
    {
      if (errno == ENOSYS)
	{
	  /* Silently ignore this test.  */
	  error (0, 0, "openat64 is not supported");
	}
      else
	error (EXIT_FAILURE, errno, "openat64 failed to open big file");
    }
  else
    {
      ret = close (fd2);

      if (ret == -1)
	error (EXIT_FAILURE, errno, "error closing file");
    }

  test_ftello ();

  return 0;
}
