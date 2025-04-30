/* Tests of *printf for very large strings.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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

#include <array_length.h>
#include <locale.h>
#include <mcheck.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <libc-diag.h>


const char *locs[] =
{
  "C", "de_DE.ISO-8859-1", "de_DE.UTF-8", "ja_JP.EUC-JP"
};

char large[50000];

static int
do_test (void)
{
  char buf[25];
  size_t i;
  int res = 0;
  int fd;

  mtrace ();

  strcpy (buf, "/tmp/test-vfprintfXXXXXX");
  fd = mkstemp (buf);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      exit (1);
    }
  unlink (buf);

  for (i = 0; i < array_length (locs); ++i)
    {
      FILE *fp;
      struct stat st;
      int fd2;

      setlocale (LC_ALL, locs[i]);

      memset (large, '\1', sizeof (large));
      large[sizeof (large) - 1] = '\0';

      fd2 = dup (fd);
      if (fd2 == -1)
	{
	  printf ("cannot dup for locale %s: %m\n",
		  setlocale (LC_ALL, NULL));
	  exit (1);
	}

      if (ftruncate (fd2, 0) != 0)
	{
	  printf ("cannot truncate file for locale %s: %m\n",
		  setlocale (LC_ALL, NULL));
	  exit (1);
	}

      fp = fdopen (fd2, "a");
      if (fp == NULL)
	{
	  printf ("cannot create FILE for locale %s: %m\n",
		  setlocale (LC_ALL, NULL));
	  exit (1);
	}

      fprintf (fp, "%s", large);
      fprintf (fp, "%.*s", 30000, large);
      large[20000] = '\0';
      /* We're testing a large format string here and need to generate it
         to avoid this source file being ridiculous.  So disable the warning
         about a generated format string.  */
      DIAG_PUSH_NEEDS_COMMENT;
      DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat-security");
      fprintf (fp, large);
      DIAG_POP_NEEDS_COMMENT;
      fprintf (fp, "%-1.300000000s", "hello");

      if (fflush (fp) != 0 || ferror (fp) != 0 || fclose (fp) != 0)
	{
	  printf ("write error for locale %s: %m\n",
		  setlocale (LC_ALL, NULL));
	  exit (1);
	}

      if (fstat (fd, &st) != 0)
	{
	  printf ("cannot stat for locale %s: %m\n",
		  setlocale (LC_ALL, NULL));
	  exit (1);
	}
      else if (st.st_size != 50000 + 30000 + 19999 + 5)
	{
	  printf ("file size incorrect for locale %s: %jd instead of %jd\n",
		  setlocale (LC_ALL, NULL),
		  (intmax_t) st.st_size,
		  (intmax_t) 50000 + 30000 + 19999 + 5);
	  res = 1;
	}
      else
	printf ("locale %s OK\n", setlocale (LC_ALL, NULL));
    }

  close (fd);

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
