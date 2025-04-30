/* POSIX regex testsuite from IEEE 2003.2.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <regex.h>
#include <stdio.h>
#include <string.h>

/* Data structure to describe the tests.  */
struct test
{
  int start;
  int end;
  const char *reg;
  const char *str;
  int options;
} tests[] =
{
#include "ptestcases.h"
};


int
main (int argc, char *argv[])
{
  size_t cnt;
  int errors = 0;

  for (cnt = 0; cnt < sizeof (tests) / sizeof (tests[0]); ++cnt)
    if (tests[cnt].str == NULL)
      {
	printf ("\n%s\n%.*s\n", tests[cnt].reg,
		(int) strlen (tests[cnt].reg),
		"-----------------------------------------------------");
      }
    else if (tests[cnt].reg == NULL)
	printf ("!!! %s\n", tests[cnt].str);
    else
      {
	regex_t re;
	regmatch_t match[20];
	int err;

	printf ("regexp: \"%s\", string: \"%s\" -> ", tests[cnt].reg,
		tests[cnt].str);

	/* Compile the expression.  */
	err = regcomp (&re, tests[cnt].reg, tests[cnt].options);
	if (err != 0)
	  {
	    if (tests[cnt].start == -2)
	      puts ("compiling failed, OK");
	    else
	      {
		char buf[100];
		regerror (err, &re, buf, sizeof (buf));
		printf ("FAIL: %s\n", buf);
		++errors;
	      }

	    continue;
	  }
	else if (tests[cnt].start == -2)
	  {
	    puts ("compiling suceeds, FAIL");
	    errors++;
	    continue;
	  }

	/* Run the actual test.  */
	err = regexec (&re, tests[cnt].str, 20, match, 0);

	if (err != 0)
	  {
	    if (tests[cnt].start == -1)
	      puts ("no match, OK");
	    else
	      {
		puts ("no match, FAIL");
		++errors;
	      }
	  }
	else
	  {
	    if (match[0].rm_so == 0 && tests[cnt].start == 0
		&& match[0].rm_eo == 0 && tests[cnt].end == 0)
	      puts ("match, OK");
	    else if (match[0].rm_so + 1 == tests[cnt].start
		     && match[0].rm_eo == tests[cnt].end)
	      puts ("match, OK");
	    else
	      {
		printf ("wrong match (%d to %d): FAIL\n",
			match[0].rm_so, match[0].rm_eo);
		++errors;
	      }
	  }

	/* Free all resources.  */
	regfree (&re);
      }

  printf ("\n%Zu tests, %d errors\n", cnt, errors);

  return errors != 0;
}
