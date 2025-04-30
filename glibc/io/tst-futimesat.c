/* futimesat basic tests.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <support/test-driver.h>
#include <support/temp_file.h>

#ifndef struct_stat
# define struct_stat struct stat64
# define fstat       fstat64
# define fstatat     fstatat64
#endif

static int dir_fd;

static void
prepare (int argc, char *argv[])
{
  size_t test_dir_len = strlen (test_dir);
  static const char dir_name[] = "/tst-futimesat.XXXXXX";

  size_t dirbuflen = test_dir_len + sizeof (dir_name);
  char *dirbuf = malloc (dirbuflen);
  if (dirbuf == NULL)
    {
      puts ("out of memory");
      exit (1);
    }

  snprintf (dirbuf, dirbuflen, "%s%s", test_dir, dir_name);
  if (mkdtemp (dirbuf) == NULL)
    {
      puts ("cannot create temporary directory");
      exit (1);
    }

  add_temp_file (dirbuf);

  dir_fd = open (dirbuf, O_RDONLY | O_DIRECTORY);
  if (dir_fd == -1)
    {
      puts ("cannot open directory");
      exit (1);
    }
}
#define PREPARE prepare

static int
do_test (void)
{
  /* fdopendir takes over the descriptor, make a copy.  */
  int dupfd = dup (dir_fd);
  if (dupfd == -1)
    {
      puts ("dup failed");
      return 1;
    }
  if (lseek (dupfd, 0, SEEK_SET) != 0)
    {
      puts ("1st lseek failed");
      return 1;
    }

  /* The directory should be empty safe the . and .. files.  */
  DIR *dir = fdopendir (dupfd);
  if (dir == NULL)
    {
      puts ("fdopendir failed");
      return 1;
    }
  struct dirent64 *d;
  while ((d = readdir64 (dir)) != NULL)
    if (strcmp (d->d_name, ".") != 0 && strcmp (d->d_name, "..") != 0)
      {
	printf ("temp directory contains file \"%s\"\n", d->d_name);
	return 1;
      }
  closedir (dir);

  /* Try to create a file.  */
  int fd = openat (dir_fd, "some-file", O_CREAT|O_RDWR|O_EXCL, 0666);
  if (fd == -1)
    {
      if (errno == ENOSYS)
	{
	  puts ("*at functions not supported");
	  return 0;
	}

      puts ("file creation failed");
      return 1;
    }
  write (fd, "hello", 5);
  puts ("file created");

  struct_stat st1;
  if (fstat (fd, &st1) != 0)
    {
      puts ("fstat64 failed");
      return 1;
    }

  close (fd);

  struct timeval tv[2];
  tv[0].tv_sec = st1.st_atime + 1;
  tv[0].tv_usec = 0;
  tv[1].tv_sec = st1.st_mtime + 1;
  tv[1].tv_usec = 0;
  if (futimesat (dir_fd, "some-file", tv) != 0)
    {
      puts ("futimesat failed");
      return 1;
    }

  struct_stat st2;
  if (fstatat (dir_fd, "some-file", &st2, 0) != 0)
    {
      puts ("fstatat64 failed");
      return 1;
    }

  if (st2.st_mtime != tv[1].tv_sec
#ifdef _STATBUF_ST_NSEC
      || st2.st_mtim.tv_nsec != 0
#endif
      )
    {
      puts ("stat shows different mtime");
      return 1;
    }


  if (unlinkat (dir_fd, "some-file", 0) != 0)
    {
      puts ("unlinkat failed");
      return 1;
    }

  close (dir_fd);

  return 0;
}

#include <support/test-driver.c>
