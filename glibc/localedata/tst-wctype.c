/* Test program for iswctype() function in ja_JP locale.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <error.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <wctype.h>

static int
do_test (void)
{
  wctype_t wct;
  wchar_t buf[1000];
  int result = 1;

  setlocale (LC_ALL, "");
  wprintf (L"locale = %s\n", setlocale (LC_CTYPE, NULL));

  wct = wctype ("jhira");
  if (wct == 0)
    error (EXIT_FAILURE, 0, "jhira: no such character class");

  if (fgetws (buf, sizeof (buf) / sizeof (buf[0]), stdin) != NULL)
    {
      int n;

      wprintf (L"buf[] = \"%ls\"\n", buf);

      result = 0;

      for (n = 0; buf[n] != L'\0'; ++n)
	{
	  wprintf (L"jhira(U%04lx = %lc) = %d\n", (long) buf[n], buf[n],
		   iswctype (buf[n], wct));
	  result |= ((buf[n] < 0xff && iswctype (buf[n], wct))
		     || (buf[n] > 0xff && !iswctype (buf[n], wct)));
	}
    }

  wct = wctype ("jkata");
  if (wct == 0)
    error (EXIT_FAILURE, 0, "jkata: no such character class");

  if (fgetws (buf, sizeof (buf) / sizeof (buf[0]), stdin) != NULL)
    {
      int n;

      wprintf (L"buf[] = \"%ls\"\n", buf);

      result = 0;

      for (n = 0; buf[n] != L'\0'; ++n)
	{
	  wprintf (L"jkata(U%04lx = %lc) = %d\n", (long) buf[n], buf[n],
		   iswctype (buf[n], wct));
	  result |= ((buf[n] < 0xff && iswctype (buf[n], wct))
		     || (buf[n] > 0xff && !iswctype (buf[n], wct)));
	}
    }

  wct = wctype ("jdigit");
  if (wct == 0)
    error (EXIT_FAILURE, 0, "jdigit: no such character class");

  if (fgetws (buf, sizeof (buf) / sizeof (buf[0]), stdin) != NULL)
    {
      int n;

      wprintf (L"buf[] = \"%ls\"\n", buf);

      result = 0;

      for (n = 0; buf[n] != L'\0'; ++n)
	{
	  wprintf (L"jdigit(U%04lx = %lc) = %d\n", (long) buf[n], buf[n],
		   iswctype (buf[n], wct));
	  result |= ((buf[n] < 0xff && iswctype (buf[n], wct))
		     || (buf[n] > 0xff && !iswctype (buf[n], wct)));
	}
    }

  wct = wctype ("jspace");
  if (wct == 0)
    error (EXIT_FAILURE, 0, "jspace: no such character class");

  if (fgetws (buf, sizeof (buf) / sizeof (buf[0]), stdin) != NULL)
    {
      int n;

      wprintf (L"buf[] = \"%ls\"\n", buf);

      result = 0;

      for (n = 0; buf[n] != L'\0'; ++n)
	{
	  wprintf (L"jspace(U%04lx = %lc) = %d\n", (long) buf[n], buf[n],
		   iswctype (buf[n], wct));
	  result |= ((buf[n] < 0xff && iswctype (buf[n], wct))
		     || (buf[n] > 0xff && !iswctype (buf[n], wct)));
	}
    }

  wct = wctype ("jkanji");
  if (wct == 0)
    error (EXIT_FAILURE, 0, "jkanji: no such character class");

  if (fgetws (buf, sizeof (buf) / sizeof (buf[0]), stdin) != NULL)
    {
      int n;

      wprintf (L"buf[] = \"%ls\"\n", buf);

      result = 0;

      for (n = 0; buf[n] != L'\0'; ++n)
	{
	  wprintf (L"jkanji(U%04lx = %lc) = %d\n", (long) buf[n], buf[n],
		   iswctype (buf[n], wct));
	  result |= ((buf[n] < 0xff && iswctype (buf[n], wct))
		     || (buf[n] > 0xff && !iswctype (buf[n], wct)));
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
