/* Test that values of pathconf and fpathconf are consistent for a file.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>


static void prepare (void);
#define PREPARE(argc, argv) prepare ()

static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"

static int dir_fd;
static char *dirbuf;

static void
prepare (void)
{
  size_t test_dir_len = strlen (test_dir);
  static const char dir_name[] = "/tst-pathconf.XXXXXX";

  size_t dirbuflen = test_dir_len + sizeof (dir_name);
  dirbuf = xmalloc (dirbuflen);

  snprintf (dirbuf, dirbuflen, "%s%s", test_dir, dir_name);
  if (mkdtemp (dirbuf) == NULL)
    {
      printf ("Cannot create temporary directory: %s\n", strerror (errno));
      exit (1);
    }

  add_temp_file (dirbuf);

  dir_fd = open (dirbuf, O_RDONLY);
  if (dir_fd == -1)
    {
      printf ("Cannot open directory: %s\n", strerror (errno));
      exit (1);
    }
}


static int
do_test (void)
{
  int ret = 0;
  static const char *fifo_name = "some-fifo";

  size_t filenamelen = strlen (dirbuf) + strlen (fifo_name) + 2;
  char *filename = xmalloc (filenamelen);

  snprintf (filename, filenamelen, "%s/%s", dirbuf, fifo_name);

  /* Create a fifo in the directory.  */
  int e = mkfifo (filename, 0777);
  if (e == -1)
    {
      printf ("fifo creation failed (%s)\n", strerror (errno));
      ret = 1;
      goto out_nofifo;
    }

  long dir_pathconf = pathconf (dirbuf, _PC_PIPE_BUF);

  if (dir_pathconf < 0)
    {
      printf ("pathconf on directory failed: %s\n", strerror (errno));
      ret = 1;
      goto out_nofifo;
    }

  long fifo_pathconf = pathconf (filename, _PC_PIPE_BUF);

  if (fifo_pathconf < 0)
    {
      printf ("pathconf on file failed: %s\n", strerror (errno));
      ret = 1;
      goto out_nofifo;
    }

  int fifo = open (filename, O_RDONLY | O_NONBLOCK);

  if (fifo < 0)
    {
      printf ("fifo open failed (%s)\n", strerror (errno));
      ret = 1;
      goto out_nofifo;
    }

  long dir_fpathconf = fpathconf (dir_fd, _PC_PIPE_BUF);

  if (dir_fpathconf < 0)
    {
      printf ("fpathconf on directory failed: %s\n", strerror (errno));
      ret = 1;
      goto out;
    }

  long fifo_fpathconf = fpathconf (fifo, _PC_PIPE_BUF);

  if (fifo_fpathconf < 0)
    {
      printf ("fpathconf on file failed: %s\n", strerror (errno));
      ret = 1;
      goto out;
    }

  if (fifo_pathconf != fifo_fpathconf)
    {
      printf ("fifo pathconf (%ld) != fifo fpathconf (%ld)\n", fifo_pathconf,
	      fifo_fpathconf);
      ret = 1;
      goto out;
    }

  if (dir_pathconf != fifo_pathconf)
    {
      printf ("directory pathconf (%ld) != fifo pathconf (%ld)\n",
	      dir_pathconf, fifo_pathconf);
      ret = 1;
      goto out;
    }

  if (dir_fpathconf != fifo_fpathconf)
    {
      printf ("directory fpathconf (%ld) != fifo fpathconf (%ld)\n",
	      dir_fpathconf, fifo_fpathconf);
      ret = 1;
      goto out;
    }

out:
  close (fifo);
out_nofifo:
  close (dir_fd);

  if (unlink (filename) != 0)
    {
      printf ("Could not remove fifo (%s)\n", strerror (errno));
      ret = 1;
    }

  return ret;
}
