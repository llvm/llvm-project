/* Test program for character classes and mappings.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1999.

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

#include <ctype.h>
#include <locale.h>
#include <wchar.h>


int
main (void)
{
  const char lower[] = "abcdefghijklmnopqrstuvwxyz";
  const char upper[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
#define LEN (sizeof (upper) - 1)
  const wchar_t wlower[] = L"abcdefghijklmnopqrstuvwxyz";
  const wchar_t wupper[] = L"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  int i;
  int result = 0;

  setlocale (LC_ALL, "test6");

  for (i = 0; i < LEN; ++i)
    {
      /* Test basic table handling (basic == not more than 256 characters).
	 The charmaps swaps the normal lower-upper case meaning of the
	 ASCII characters used in the source code while the Unicode mapping
	 in the repertoire map has the normal correspondents.  This test
	 shows the independence of the tables for `char' and `wchar_t'
	 characters.  */

      if (islower (lower[i]))
	{
	  printf ("islower ('%c') false\n", lower[i]);
	  result = 1;
	}
      if (! isupper (lower[i]))
	{
	  printf ("isupper ('%c') false\n", lower[i]);
	  result = 1;
	}

      if (! islower (upper[i]))
	{
	  printf ("islower ('%c') false\n", upper[i]);
	  result = 1;
	}
      if (isupper (upper[i]))
	{
	  printf ("isupper ('%c') false\n", upper[i]);
	  result = 1;
	}

      if (toupper (lower[i]) != lower[i])
	{
	  printf ("toupper ('%c') false\n", lower[i]);
	  result = 1;
	}
      if (tolower (lower[i]) != upper[i])
	{
	  printf ("tolower ('%c') false\n", lower[i]);
	  result = 1;
	}

      if (tolower (upper[i]) != upper[i])
	{
	  printf ("tolower ('%c') false\n", upper[i]);
	  result = 1;
	}
      if (toupper (upper[i]) != lower[i])
	{
	  printf ("toupper ('%c') false\n", upper[i]);
	  result = 1;
	}

      if (iswlower (wupper[i]))
	{
	  printf ("iswlower (L'%c') false\n", upper[i]);
	  result = 1;
	}
      if (! iswupper (wupper[i]))
	{
	  printf ("iswupper (L'%c') false\n", upper[i]);
	  result = 1;
	}

      if (iswupper (wlower[i]))
	{
	  printf ("iswupper (L'%c') false\n", lower[i]);
	  result = 1;
	}
      if (! iswlower (wlower[i]))
	{
	  printf ("iswlower (L'%c') false\n", lower[i]);
	  result = 1;
	}

      if (towupper (wlower[i]) != wupper[i])
	{
	  printf ("towupper ('%c') false\n", lower[i]);
	  result = 1;
	}
      if (towlower (wlower[i]) != wlower[i])
	{
	  printf ("towlower ('%c') false\n", lower[i]);
	  result = 1;
	}

      if (towlower (wupper[i]) != wlower[i])
	{
	  printf ("towlower ('%c') false\n", upper[i]);
	  result = 1;
	}
      if (towupper (wupper[i]) != wupper[i])
	{
	  printf ("towupper ('%c') false\n", upper[i]);
	  result = 1;
	}
    }

  return result;
}
