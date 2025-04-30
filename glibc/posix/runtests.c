/***********************************************************

Copyright 1995 by Tom Lord

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in
supporting documentation, and that the name of the copyright holder not be
used in advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

Tom Lord DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
EVENT SHALL TOM LORD BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

******************************************************************/



#include <sys/types.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



struct a_test
{
  int expected;
  const char * pattern;
  const char * data;
};

static const struct a_test the_tests[] =
{
#include "testcases.h"
  {-1, 0, 0}
};




static int
run_a_test (int id, const struct a_test * t)
{
  static const char * last_pattern = 0;
  static regex_t r;
  int err;
  char errmsg[100];
  int x;
  regmatch_t regs[10];

  if (!last_pattern || strcmp (last_pattern, t->pattern))
    {
      if (last_pattern)
	regfree (&r);
      last_pattern = t->pattern;
      err = regcomp (&r, t->pattern, REG_EXTENDED);
      if (err)
	{
	  if (t->expected == 2)
	    {
	      puts (" OK.");
	      return 0;
	    }
	  if (last_pattern)
	    regfree (&r);
	  last_pattern = NULL;
	  regerror (err, &r, errmsg, 100);
	  printf (" FAIL: %s.\n", errmsg);
	  return 1;
	}
      else if (t->expected == 2)
	{
	  printf ("test %d\n", id);
	  printf ("pattern \"%s\" successfull compilation not expected\n",
		  t->pattern);
	  return 1;
	}
    }

  err = regexec (&r, t->data, 10, regs, 0);

  if (err != t->expected)
    {
      printf ("test %d\n", id);
      printf ("pattern \"%s\" data \"%s\" wanted %d got %d\n",
	      t->pattern, t->data, t->expected, err);
      for (x = 0; x < 10; ++x)
	printf ("reg %d == (%d, %d) %.*s\n",
		x,
		regs[x].rm_so,
		regs[x].rm_eo,
		regs[x].rm_eo - regs[x].rm_so,
		t->data + regs[x].rm_so);
      return 1;
    }
  puts (" OK.");
  return 0;
}



int
main (int argc, char * argv[])
{
  int x;
  int lo;
  int hi;
  int res = 0;

  lo = 0;
  hi = (sizeof (the_tests) / sizeof (the_tests[0])) - 1;

  if (argc > 1)
    {
      lo = atoi (argv[1]);
      hi = lo + 1;

      if (argc > 2)
	hi = atoi (argv[2]);
    }

  for (x = lo; x < hi; ++x)
    {
      printf ("#%d:", x);
      res |= run_a_test (x, &the_tests[x]);
    }
  return res != 0;
}
