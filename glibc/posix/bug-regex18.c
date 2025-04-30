/* Turkish regular expression tests.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2002.

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

#include <sys/types.h>
#include <mcheck.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

/* Tests supposed to match.  */
struct
{
  const char *pattern;
  const char *string;
  int flags, nmatch;
  regmatch_t rm[5];
} tests[] = {
  /* \xc4\xb0		LATIN CAPITAL LETTER I WITH DOT ABOVE
     \xc4\xb1		LATIN SMALL LETTER DOTLESS I
     \xe2\x80\x94	EM DASH  */
  { "\xc4\xb0I*\xc4\xb1$", "aBi\xc4\xb1\xc4\xb1I", REG_ICASE, 2,
    { { 2, 8 }, { -1, -1 } } },
  { "[\xc4\xb0x]I*\xc4\xb1$", "aBi\xc4\xb1\xc4\xb1I", REG_ICASE, 2,
    { { 2, 8 }, { -1, -1 } } },
  { "[^x]I*\xc4\xb1$", "aBi\xc4\xb1\xc4\xb1I", REG_ICASE, 2,
    { { 2, 8 }, { -1, -1 } } },
  { "([[:alpha:]]i[[:xdigit:]])(\xc4\xb1*)(\xc4\xb0{2})",
    "\xe2\x80\x94\xc4\xb1\xc4\xb0""fIi\xc4\xb0ii", REG_ICASE | REG_EXTENDED,
    4, { { 3, 12 }, { 3, 8 }, { 8, 9 }, { 9, 12 } } },
  { "\xc4\xb1i(i)*()(\\s\xc4\xb0|\\SI)", "SIi\xc4\xb0\xc4\xb0 is",
    REG_ICASE | REG_EXTENDED, 4, { { 1, 9 }, { 5, 7 }, { 7, 7 }, { 7, 9 } } },
  { "\xc4\xb1i(i)*()(\\s\xc4\xb0|\\SI)", "\xc4\xb1\xc4\xb0\xc4\xb0iJ\xc4\xb1",
    REG_ICASE | REG_EXTENDED, 4,
    { { 0, 10 }, { 6, 7 }, { 7, 7 }, { 7, 10 } } },
};

int
main (void)
{
  regex_t re;
  regmatch_t rm[5];
  size_t i;
  int n, ret = 0;

  setlocale (LC_ALL, "tr_TR.UTF-8");
  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      n = regcomp (&re, tests[i].pattern, tests[i].flags);
      if (n != 0)
	{
	  char buf[500];
	  regerror (n, &re, buf, sizeof (buf));
	  printf ("regcomp %zd failed: %s\n", i, buf);
	  ret = 1;
	  continue;
	}

      if (regexec (&re, tests[i].string, tests[i].nmatch, rm, 0))
	{
	  printf ("regexec %zd failed\n", i);
	  ret = 1;
	  regfree (&re);
	  continue;
	}

      for (n = 0; n < tests[i].nmatch; ++n)
	if (rm[n].rm_so != tests[i].rm[n].rm_so
              || rm[n].rm_eo != tests[i].rm[n].rm_eo)
	  {
	    if (tests[i].rm[n].rm_so == -1 && tests[i].rm[n].rm_eo == -1)
	      break;
	    printf ("regexec match failure rm[%d] %d..%d\n",
		    n, rm[n].rm_so, rm[n].rm_eo);
	    ret = 1;
	    break;
	  }

      regfree (&re);
    }

  return ret;
}
