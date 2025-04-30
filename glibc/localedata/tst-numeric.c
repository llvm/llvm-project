/* Testing the implementation of LC_NUMERIC and snprintf().
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Petter Reinholdtsen <pere@hungry.com>, 2003

   Based on tst-fmon.c by Jochen Hein <jochen.hein@delphi.central.de>, 1997.

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

#include <stdio.h>
#include <locale.h>
#include <string.h>
#include <stdlib.h>

/*
  test-numeric gets called with three parameters:
   - the locale
   - the format-string to be used
   - the actual number to be formatted
   - the expected string
   If the test passes, test-numeric terminates with returncode 0,
   otherwise with 1
*/
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define EXIT_SETLOCALE 2
#define EXIT_SNPRINTF 3

int
main (int argc, char *argv[])
{
  char s[200];
  double val;

  /* Make sure to read the value before setting of the locale, as
     strtod() is locale-dependent. */
  val = strtod (argv[3], NULL);

  if (setlocale (LC_ALL, argv[1]) == NULL)
    {
      fprintf (stderr, "setlocale(LC_ALL, \"%s\"): %m\n", argv[1]);
      exit (EXIT_SETLOCALE);
    }

  if (snprintf (s, sizeof (s), argv[2], val) == -1)
    {
      perror ("snprintf");
      exit (EXIT_SNPRINTF);
    }

  if (strcmp (s, argv[4]) != 0)
    {
      printf ("\
locale: \"%s\", format: \"%s\", expected: \"%s\", got: \"%s\" => %s\n",
	      argv[1], argv[2], argv[4], s,
	      strcmp (s, argv[4]) != 0 ? "false" : "correct");
      exit (EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}
