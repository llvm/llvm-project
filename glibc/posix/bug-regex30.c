/* Russian regular expression tests.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Paolo Bonzini <pbonzini@redhat.com>, 2009.

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
  /* U+0413	\xd0\x93	CYRILLIC CAPITAL LETTER GHE
     U+0420	\xd0\xa0        CYRILLIC CAPITAL LETTER ER
     U+0430	\xd0\xb0	CYRILLIC SMALL LETTER A
     U+0433	\xd0\xb3	CYRILLIC SMALL LETTER GHE
     U+0440	\xd1\x80	CYRILLIC SMALL LETTER ER
     U+044F	\xd1\x8f	CYRILLIC SMALL LETTER YA */
  { "[\xd0\xb0-\xd1\x8f]", "\xd0\xb3", 0, 1,
    { { 0, 2 } } },
  { "[\xd0\xb0-\xd1\x8f]", "\xd0\x93", REG_ICASE, 1,
    { { 0, 2 } } },
  { "[\xd1\x80-\xd1\x8f]", "\xd0\xa0", REG_ICASE, 1,
    { { 0, 2 } } },
};


static int
do_test (void)
{
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  int ret = 0;

  for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      regex_t re;
      regmatch_t rm[5];
      int n = regcomp (&re, tests[i].pattern, tests[i].flags);
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

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
