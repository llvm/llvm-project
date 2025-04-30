/* Regular expression tests.
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

/* Tests supposed to not match.  */
struct
{
  const char *pattern;
  const char *string;
  int flags, nmatch;
} tests[] = {
  { "^<\\([^~]*\\)\\([^~]\\)[^~]*~\\1\\(.\\).*|=.*\\3.*\\2",
    "<,.8~2,~so-|=-~.0,123456789<><", REG_NOSUB, 0 },
  /* In ERE, all carets must be treated as anchors.  */
  { "a^b", "a^b", REG_EXTENDED, 0 }
};

int
main (void)
{
  regex_t re;
  regmatch_t rm[4];
  size_t i;
  int n, ret = 0;

  mtrace ();

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

      if (! regexec (&re, tests[i].string, tests[i].nmatch,
		     tests[i].nmatch ? rm : NULL, 0))
	{
	  printf ("regexec %zd incorrectly matched\n", i);
	  ret = 1;
	}

      regfree (&re);
    }

  return ret;
}
