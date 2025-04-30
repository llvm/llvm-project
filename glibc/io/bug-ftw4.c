/* Test if ftw function doesn't leak fds.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <ftw.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int cb_called;

static int
cb (const char *name, const struct stat64 *st, int type)
{
  return cb_called++ & 1;
}

int
main (void)
{
  char name[32] = "/tmp/ftwXXXXXX", *p;
  int ret, i, result = 0, fd, fd1, fd2;

  if (mkdtemp (name) == NULL)
    {
      printf ("Couldn't make temporary directory: %m\n");
      exit (EXIT_FAILURE);
    }
  p = strchr (name, '\0');
  strcpy (p, "/1");
  if (mkdir (name, 0755) < 0)
    {
      printf ("Couldn't make temporary subdirectory: %m\n");
      exit (EXIT_FAILURE);
    }
  *p = '\0';

  ret = ftw64 (name, cb, 20);
  if (ret != 1)
    {
      printf ("ftw64 returned %d instead of 1", ret);
      result = 1;
    }

  fd = open (name, O_RDONLY);
  if (fd < 0)
    {
      printf ("open failed: %m\n");
      result = 1;
    }
  fd1 = open (name, O_RDONLY);
  if (fd1 < 0)
    {
      printf ("open failed: %m\n");
      result = 1;
    }
  else
    close (fd1);
  if (fd >= 0)
    close (fd);

  for (i = 0; i < 128; ++i)
    {
      ret = ftw64 (name, cb, 20);
      if (ret != 1)
	{
	  printf ("ftw64 returned %d instead of 1", ret);
	  result = 1;
	}
    }

  fd = open (name, O_RDONLY);
  if (fd < 0)
    {
      printf ("open failed: %m\n");
      result = 1;
    }
  fd2 = open (name, O_RDONLY);
  if (fd2 < 0)
    {
      printf ("open failed: %m\n");
      result = 1;
    }
  else
    close (fd2);
  if (fd >= 0)
    close (fd);

  if (fd2 >= fd1 + 128)
    {
      printf ("ftw64 leaking fds: %d -> %d\n", fd1, fd2);
      result = 1;
    }

  if (cb_called != 129 * 2)
    {
      printf ("callback called %d times\n", cb_called);
      result = 1;
    }

  strcpy (p, "/1");
  rmdir (name);
  *p = '\0';
  rmdir (name);
  return result;
}
