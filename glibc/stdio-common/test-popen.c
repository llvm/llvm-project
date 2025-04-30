/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

static void
write_data (FILE *stream)
{
  int i;
  for (i=0; i<100; i++)
    fprintf (stream, "%d\n", i);
  if (ferror (stream))
    {
      fprintf (stderr, "Output to stream failed.\n");
      exit (1);
    }
}

static void
read_data (FILE *stream)
{
  int i, j;

  for (i=0; i<100; i++)
    {
      if (fscanf (stream, "%d\n", &j) != 1 || j != i)
	{
	  if (ferror (stream))
	    perror ("fscanf");
	  puts ("Test FAILED!");
	  exit (1);
	}
    }
}

static int
do_test (void)
{
  FILE *output, *input;
  int wstatus, rstatus;

  /* We must remove this entry to assure the `cat' binary does not use
     the perhaps incompatible new shared libraries.  */
  unsetenv ("LD_LIBRARY_PATH");

  output = popen ("/bin/cat >" OBJPFX "tstpopen.tmp", "w");
  if (output == NULL)
    {
      perror ("popen");
      puts ("Test FAILED!");
      exit (1);
    }
  write_data (output);
  wstatus = pclose (output);
  printf ("writing pclose returned %d\n", wstatus);
  input = popen ("/bin/cat " OBJPFX "tstpopen.tmp", "r");
  if (input == NULL)
    {
      perror (OBJPFX "tstpopen.tmp");
      puts ("Test FAILED!");
      exit (1);
    }
  read_data (input);
  rstatus = pclose (input);
  printf ("reading pclose returned %d\n", rstatus);

  remove (OBJPFX "tstpopen.tmp");

  errno = 0;
  output = popen ("/bin/cat", "m");
  if (output != NULL)
    {
      puts ("popen called with illegal mode does not return NULL");
      puts ("Test FAILED!");
      exit (1);
    }
  if (errno != EINVAL)
    {
      puts ("popen called with illegal mode does not set errno to EINVAL");
      puts ("Test FAILED!");
      exit (1);
    }

  puts (wstatus | rstatus  ? "Test FAILED!" : "Test succeeded.");
  return (wstatus | rstatus);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
