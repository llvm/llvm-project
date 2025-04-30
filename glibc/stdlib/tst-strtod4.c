#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NNBSP "\xe2\x80\xaf"

static const struct
{
  const char *in;
  const char *out;
  double expected;
} tests[] =
  {
    { "000"NNBSP"000"NNBSP"000", "", 0.0 },
    { "1"NNBSP"000"NNBSP"000,5x", "x", 1000000.5 }
  };
#define NTESTS (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  if (setlocale (LC_ALL, "cs_CZ.UTF-8") == NULL)
    {
      puts ("could not set locale");
      return 1;
    }

  int status = 0;

  for (int i = 0; i < NTESTS; ++i)
    {
      char *ep;
      double r = __strtod_internal (tests[i].in, &ep, 1);

      if (strcmp (ep, tests[i].out) != 0)
	{
	  printf ("%d: got rest string \"%s\", expected \"%s\"\n",
		  i, ep, tests[i].out);
	  status = 1;
	}

      if (r != tests[i].expected)
	{
	  printf ("%d: got wrong results %g, expected %g\n",
		  i, r, tests[i].expected);
	  status = 1;
	}
    }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
