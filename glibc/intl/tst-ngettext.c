/* Test of the ngettext functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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
#include <libintl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int
do_test (void)
{
  const char *strs[2] = { "singular", "plural" };
  unsigned long int i;
  int res = 0;

  /* We don't want any translation here.  */
  setenv ("LANGUAGE", "C", 1);
  unsetenv ("OUTPUT_CHARSET");

  for (i = 0; i < 30; ++i)
    {
      char *tr;

      tr = ngettext (strs[0], strs[1], i);
#define TEST \
      do								      \
	if (tr != strs[i != 1])						      \
	  {								      \
	    if (strcmp (tr, strs[i != 1]) == 0)				      \
	      printf ("%lu: correct string, wrong pointer (%s)\n", i, tr);    \
	    else							      \
	      printf ("%lu: wrong result (%s)\n", i, tr);		      \
	    res = 1;							      \
	  }								      \
      while (0)
      TEST;

      tr = dngettext ("messages", strs[0], strs[1], i);
      TEST;

      tr = dcngettext ("messages", strs[0], strs[1], i, LC_MESSAGES);
      TEST;
    }

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
