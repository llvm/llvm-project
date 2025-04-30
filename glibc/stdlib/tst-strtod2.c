#include <stdio.h>
#include <stdlib.h>

struct test
{
  const char *str;
  double result;
  size_t offset;
} tests[] =
{
  { "0xy", 0.0, 1 },
  { "0x.y", 0.0, 1 },
  { "0x0.y", 0.0, 4 },
  { "0x.0y", 0.0, 4 },
  { ".y", 0.0, 0 },
  { "0.y", 0.0, 2 },
  { ".0y", 0.0, 2 }
};

static int
do_test (void)
{
  int status = 0;
  for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      char *ep;
      double r = strtod (tests[i].str, &ep);
      if (r != tests[i].result)
	{
	  printf ("test %zu r = %g, expect %g\n", i, r, tests[i].result);
	  status = 1;
	}
      if (ep != tests[i].str + tests[i].offset)
	{
	  printf ("test %zu strtod parsed %tu characters, expected %zu\n",
		  i, ep - tests[i].str, tests[i].offset);
	  status = 1;
	}
    }
  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
