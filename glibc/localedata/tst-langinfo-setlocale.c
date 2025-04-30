/* Test program for setlocale() + nl_langinfo() functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <langinfo.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>

/* Return 0 if the test passed, 1 for failed.  */
static int
test_locale (char *locale, char *paramstr, int param, char *expected)
{
  char *actual;

  printf ("LC_ALL=%s nl_langinfo(%s)", locale, paramstr);

  /* Set the locale and check whether it worked.  */
  setlocale (LC_ALL, locale);
  if (strcmp (locale, setlocale (LC_ALL, NULL)) != 0)
    {
      puts (": failed to set locale");
      return 1;
    }

  actual = nl_langinfo (param);
  printf (" = \"%s\", ", actual);

  if (strcmp (actual, expected) == 0)
    puts ("OK");
  else
    {
      printf ("FAILED (expected: %s)\n", expected);
      return 1;
    }

  return 0;
}

#include <tst-langinfo.c>
