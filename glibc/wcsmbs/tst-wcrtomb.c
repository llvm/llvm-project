/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>


static int check_ascii (const char *locname);


static int
do_test (void)
{
  int result = 0;

  /* Check mapping of ASCII range for some character sets which have
     ASCII as a subset.  For those the wide char generated must have
     the same value.  */
  setlocale (LC_ALL, "C");
  result |= check_ascii (setlocale (LC_ALL, NULL));

  setlocale (LC_ALL, "de_DE.UTF-8");
  result |= check_ascii (setlocale (LC_ALL, NULL));

  setlocale (LC_ALL, "ja_JP.EUC-JP");
  result |= check_ascii (setlocale (LC_ALL, NULL));

  return result;
}


static int
check_ascii (const char *locname)
{
  wchar_t wc;
  int res = 0;

  printf ("Testing locale \"%s\":\n", locname);

  for (wc = 0; wc <= 127; ++wc)
    {
      char buf[2 * MB_CUR_MAX];
      mbstate_t s;
      size_t n;

      memset (buf, '\xff', sizeof (buf));
      memset (&s, '\0', sizeof (s));

      n = wcrtomb (buf, wc, &s);
      if (n == (size_t) -1)
	{
	  printf ("%s: '\\x%x': encoding error\n", locname, (int) wc);
	  ++res;
	}
      else if (n == 0)
	{
	  printf ("%s: '\\x%x': 0 returned\n", locname, (int) wc);
	  ++res;
	}
      else if (n != 1)
	{
	  printf ("%s: '\\x%x': not 1 returned\n", locname, (int) wc);
	  ++res;
	}
      else if (wc != (wchar_t) buf[0])
	{
	  printf ("%s: L'\\x%x': buf[0] != '\\x%x'\n", locname, (int) wc,
		  (int) wc);
	  ++res;
	}
    }

  printf (res == 1 ? "%d error\n" : "%d errors\n", res);

  return res != 0;
}

#include <support/test-driver.c>
