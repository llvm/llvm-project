/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <wctype.h>

int
main (int argc, char *argv[])
{
  int result = 0;
  wint_t ch;


  for (ch = 0; ch < 128; ++ch)
    {
      if (iswlower (ch))
	{
	  /* Get corresponding upper case character.  */
	  wint_t up = towupper (ch);
	  /* This should have no effect.  */
	  wint_t low  = towlower (ch);

	  if ((ch != low) || (up == ch) || (up == low))
	    {
	      printf ("iswlower/towupper/towlower for character \\%x failed\n", ch);
	      result++;
	    }
	}
      if (iswupper (ch))
	{
	  /* Get corresponding lower case character.  */
	  wint_t low = towlower (ch);
	  /* This should have no effect.  */
	  wint_t up  = towupper (ch);

	  if ((ch != up) || (low == ch) || (up == low))
	    {
	      printf ("iswupper/towlower/towupper for character \\%x failed\n", ch);
	      result++;
	    }
	}
    }

  /* Finally some specific tests.  */
  ch = L'A';
  if (!iswupper (ch) || iswlower (ch))
    {
      printf ("!iswupper/iswlower (L'A') failed\n");
      result++;

    }
  ch = L'a';
  if (iswupper (ch) || !iswlower (ch))
    {
      printf ("iswupper/!iswlower (L'a') failed\n");
      result++;
    }
  if (towlower (L'A') != L'a')
    {
      printf ("towlower(L'A') failed\n");
      result++;
    }
  if (towupper (L'a') != L'A')
    {
      printf ("towupper(L'a') failed\n");
      result++;
    }

  if (result == 0)
    puts ("All test successful!");
  return result != 0;
}
