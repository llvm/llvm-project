/* Test program for user-defined character maps.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>.

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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

static int
do_test (void)
{
  char buf[30];
  wchar_t wbuf[30];
  wctrans_t t;
  wint_t wch;
  int errors = 0;
  int len;

  setlocale (LC_ALL, "");

  t = wctrans ("test");
  if (t == (wctrans_t) 0)
    {
      puts ("locale data files probably not loaded");
      exit (1);
    }

  wch = towctrans (L'A', t);
  printf ("towctrans (L'A', t) = %lc\n", wch);
  if (wch != L'B')
    errors = 1;

  wch = towctrans (L'B', t);
  printf ("towctrans (L'B', t) = %lc\n", wch);
  if (wch != L'C')
    errors = 1;

  /* Test the output digit handling.  */
  swprintf (wbuf, sizeof (wbuf) / sizeof (wbuf[0]), L"%Id", 0x499602D2);
  errors |= wcscmp (wbuf, L"bcdefghija") != 0;
  len = wcslen (wbuf);
  errors |= len != 10;
  printf ("len = %d, wbuf = L\"%ls\"\n", len, wbuf);

  snprintf (buf, sizeof buf, "%Id", 0x499602D2);
  errors |= strcmp (buf, "bcdefghija") != 0;
  len = strlen (buf);
  errors |= len != 10;
  printf ("len = %d, buf = \"%s\"\n", len, buf);

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
