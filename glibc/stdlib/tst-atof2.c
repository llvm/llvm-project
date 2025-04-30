#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static const struct
{
  const char *str;
  const char *expected;
} tests[] =
  {
    { "1e308", "1e+308" },
    { "100000000e300", "1e+308" },
    { "0x1p1023", "8.98847e+307" },
    { "0x1000p1011", "8.98847e+307" },
    { "0x1p1020", "1.12356e+307" },
    { "0x0.00001p1040", "1.12356e+307" },
    { "1e-307", "1e-307" },
    { "0.000001e-301", "1e-307" },
    { "0.0000001e-300", "1e-307" },
    { "0.00000001e-299", "1e-307" },
    { "1000000e-313", "1e-307" },
    { "10000000e-314", "1e-307" },
    { "100000000e-315", "1e-307" },
    { "0x1p-1021", "4.45015e-308" },
    { "0x1000p-1033", "4.45015e-308" },
    { "0x10000p-1037", "4.45015e-308" },
    { "0x0.001p-1009", "4.45015e-308" },
    { "0x0.0001p-1005", "4.45015e-308" },
  };
#define NTESTS (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  int status = 0;

  for (int i = 0; i < NTESTS; ++i)
    {
      char buf[100];
      snprintf (buf, sizeof (buf), "%g", atof (tests[i].str));
      if (strcmp (buf, tests[i].expected) != 0)
	{
	  printf ("%d: got \"%s\", expected \"%s\"\n",
		  i, buf, tests[i].expected);
	  status = 1;
	}
    }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
