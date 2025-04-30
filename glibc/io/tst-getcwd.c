/* Test of getcwd function.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <libc-diag.h>


#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  char thepath[4096];	/* Yes, this limits the environment this test
			   can run it but I honestly don't care about
			   people which have this problem.  */
  char *bufs[10];
  size_t lens[10];
  size_t sbs;
  size_t len, i;

  if (getcwd (thepath, sizeof thepath) == NULL)
    {
      if (errno == ERANGE)
	/* The path is too long, skip all tests.  */
	return 0;

      puts ("getcwd (thepath, sizeof thepath) failed");
      return 1;
    }
  len = strlen (thepath);

  sbs = 1;
  while (sbs < len + 1)
    sbs <<= 1;

  for (i = 0; i < 4; ++i)
    {
      lens[i] = sbs;
      bufs[i] = (char *) malloc (sbs);
    }

  /* Avoid warnings about the first argument being null when the second
     is nonzero.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (10.1, "-Wnonnull");
  bufs[i] = getcwd (NULL, sbs);
  DIAG_POP_NEEDS_COMMENT;

  lens[i] = sbs;
  if (bufs[i] == NULL)
    {
      puts ("getcwd (NULL, sbs) failed");
      return 1;
    }
  ++i;

  for (; i < 10; sbs >>= 1, ++i)
    {
      bufs[i] = (char *) malloc (MAX (1, sbs));
      lens[i] = sbs;
    }

  /* Before we test the result write something in the memory to see
     whether the allocation went right.  */
  for (i = 0; i < 10; ++i)
    if (i != 4 && bufs[i] != NULL)
      memset (bufs[i], '\xff', lens[i]);

  if (strcmp (thepath, bufs[4]) != 0)
    {
      printf ("\
getcwd (NULL, sbs) = \"%s\", getcwd (thepath, sizeof thepath) = \"%s\"\n",
	      bufs[4], thepath);
      return 1;
    }

  /* Now overwrite all buffers to see that getcwd allocated the buffer
     of right size.  */
  for (i = 0; i < 10; ++i)
    memset (bufs[i], i, lens[i]);

  for (i = 0; i < 10; ++i)
    free (bufs[i]);

  /* Test whether the function signals success despite the buffer
     being too small.
     Avoid warnings about the first argument being null when the second
     is nonzero.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (10.1, "-Wnonnull");
  if (getcwd (NULL, len) != NULL)
    {
      puts ("getcwd (NULL, len) didn't failed");
      return 1;
    }
  DIAG_POP_NEEDS_COMMENT;

  bufs[0] = malloc (len);
  bufs[1] = malloc (len);
  bufs[2] = malloc (len);
  if (bufs[1] != NULL)
    {
      if (getcwd (bufs[1], len) != NULL)
	{
	  puts ("getcwd (bufs[1], len) didn't failed");
	  return 1;
	}
      free (bufs[0]);
      free (bufs[1]);
      free (bufs[2]);
    }

  memset (thepath, '\xfe', sizeof (thepath));
  if (getcwd (thepath, len) != NULL)
    {
      puts ("getcwd (thepath, len) didn't failed");
      return 1;
    }

  for (i = len; i < sizeof thepath; ++i)
    if (thepath[i] != '\xfe')
      {
	puts ("thepath[i] != '\xfe'");
	return 1;
      }

  /* Now test handling of correctly sized buffers.
     Again. avoid warnings about the first argument being null when
     the second is nonzero.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (10.1, "-Wnonnull");
  bufs[0] = getcwd (NULL, len + 1);
  if (bufs[0] == NULL)
    {
      puts ("getcwd (NULL, len + 1) failed");
      return 1;
    }
  DIAG_POP_NEEDS_COMMENT;
  free (bufs[0]);

  memset (thepath, '\xff', sizeof thepath);
  if (getcwd (thepath, len + 1) == NULL)
    {
      puts ("getcwd (thepath, len + 1) failed");
      return 1;
    }

  for (i = len + 1; i < sizeof thepath; ++i)
    if (thepath[i] != '\xff')
      {
	printf ("thepath[%zd] != '\xff'\n", i);
	return 1;
      }

  puts ("everything OK");

  return 0;
}

#include "../test-skeleton.c"
