/* Testing the implementation of strfmon(3).
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jochen Hein <jochen.hein@delphi.central.de>, 1997.

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
#include <monetary.h>
#include <string.h>
#include <stdlib.h>

/*
  test-strfmon gets called with three parameters:
   - the locale
   - the format-string to be used
   - the actual number to be formatted
   - the expected string
   If the test passes, test-strfmon terminates with returncode 0,
   otherwise with 1
*/
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define EXIT_SETLOCALE 2
#define EXIT_STRFMON 3

int
main (int argc, char *argv[])
{
  char s[200];

  if (setlocale (LC_MONETARY, argv[1]) == NULL)
    {
      fprintf (stderr, "setlocale(LC_MONETARY, \"%s\"): %m\n", argv[1]);
      exit (EXIT_SETLOCALE);
    }

  if (strfmon (s, sizeof (s), argv[2], (double) atof (argv[3])) == -1)
    {
      perror ("strfmon");
      exit (EXIT_STRFMON);
    }

  if (strcmp (s, argv[4]) != 0)
    {
      printf ("\
Locale: \"%s\" Format: \"%s\" Value: \"%s\" Received: \"%s\" Expected: \"%s\" => %s\n",
	      argv[1], argv[2], argv[3], s, argv[4],
	      strcmp (s, argv[4]) != 0 ? "false" : "correct");
      exit (EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}
