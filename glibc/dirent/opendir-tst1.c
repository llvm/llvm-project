/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

/* Name of the FIFO.  */
char tmpname[] = "fifoXXXXXX";


/* Do the real work.  */
static int
real_test (void)
{
  DIR *dirp;

  /* This should not block for an FIFO.  */
  dirp = opendir (tmpname);

  /* Successful.  */
  if (dirp != NULL)
    {
      /* Oh, oh, how can this work?  */
      fputs ("`opendir' succeeded on a FIFO???\n", stdout);
      closedir (dirp);
      return 1;
    }

  if (errno != ENOTDIR)
    {
      fprintf (stdout, "`opendir' return error `%s' instead of `%s'\n",
	       strerror (errno), strerror (ENOTDIR));
      return 1;
    }

  return 0;
}


static int
do_test (void)
{
  int retval;

  if (mktemp (tmpname) == NULL)
    {
      perror ("mktemp");
      return 1;
    }

  /* Try to generate a FIFO.  */
  if (mknod (tmpname, 0600 | S_IFIFO, 0) < 0)
    {
      perror ("mknod");
      /* We cannot make this an error.  */
      return 0;
    }

  retval = real_test ();

  remove (tmpname);

  return retval;
}


static void
do_cleanup (void)
{
  remove (tmpname);
}
#define CLEANUP_HANDLER do_cleanup

#include <support/test-driver.c>
