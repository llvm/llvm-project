/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int
main (int argc, char **argv)
{
  static const char hello[] = "Hello, world.\n";
  static const char replace[] = "Hewwo, world.\n";
  static const size_t replace_from = 2, replace_to = 4;
  char filename[FILENAME_MAX];
  char *name = strrchr (*argv, '/');
  char buf[BUFSIZ];
  FILE *f;
  int lose = 0;

  if (name != NULL)
    ++name;
  else
    name = *argv;

  (void) sprintf (filename, OBJPFX "%s.test", name);

  f = fopen (filename, "w+");
  if (f == NULL)
    {
      perror (filename);
      exit (1);
    }

  (void) fputs (hello, f);
  rewind (f);
  (void) fgets (buf, sizeof (buf), f);
  rewind (f);
  (void) fputs (buf, f);
  rewind (f);
  {
    size_t i;
    for (i = 0; i < replace_from; ++i)
      {
	int c = getc (f);
	if (c == EOF)
	  {
	    printf ("EOF at %Zu.\n", i);
	    lose = 1;
	    break;
	  }
	else if (c != hello[i])
	  {
	    printf ("Got '%c' instead of '%c' at %Zu.\n",
		    (unsigned char) c, hello[i], i);
	    lose = 1;
	    break;
	  }
      }
  }

  {
    long int where = ftell (f);
    if (where == (long int) replace_from)
      {
	size_t i;
	for (i = replace_from; i < replace_to; ++i)
	  if (putc(replace[i], f) == EOF)
	    {
	      printf ("putc('%c') got %s at %Zu.\n",
		      replace[i], strerror (errno), i);
	      lose = 1;
	      break;
	    }
      }
    else if (where == -1L)
      {
	printf ("ftell got %s (should be at %Zu).\n",
		strerror (errno), replace_from);
	lose = 1;
      }
    else
      {
	printf ("ftell returns %lu; should be %Zu.\n", where, replace_from);
	lose = 1;
      }
  }

  if (!lose)
    {
      rewind (f);
      if (fgets (buf, sizeof (buf), f) == NULL)
	{
	  printf ("fgets got %s.\n", strerror(errno));
	  lose = 1;
	}
      else if (strcmp (buf, replace))
	{
	  printf ("Read \"%s\" instead of \"%s\".\n", buf, replace);
	  lose = 1;
	}
    }

  if (lose)
    printf ("Test FAILED!  Losing file is \"%s\".\n", filename);
  else
    {
      (void) remove (filename);
      puts ("Test succeeded.");
    }

  return lose ? EXIT_FAILURE : EXIT_SUCCESS;
}
