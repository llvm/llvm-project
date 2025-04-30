#include <array_length.h>
#include <stdio.h>
#include <string.h>

struct
{
  long double val;
  const char str[4][7];
} tests[] =
{
  { 0x0.FFFFp+0L, { "0X1P+0", "0X2P-1", "0X4P-2", "0X8P-3" } },
  { 0x0.FFFFp+1L, { "0X1P+1", "0X2P+0", "0X4P-1", "0X8P-2" } },
  { 0x0.FFFFp+2L, { "0X1P+2", "0X2P+1", "0X4P+0", "0X8P-1" } },
  { 0x0.FFFFp+3L, { "0X1P+3", "0X2P+2", "0X4P+1", "0X8P+0" } }
};

static int
do_test (void)
{
  char buf[100];
  int ret = 0;

  for (size_t i = 0; i < array_length (tests); ++i)
    {
      snprintf (buf, sizeof (buf), "%.0LA", tests[i].val);

      size_t j;
      for (j = 0; j < 4; ++j)
	if (strcmp (buf, tests[i].str[j]) == 0)
	  break;

      if (j == 4)
	{
	  printf ("%zd: got \"%s\", expected \"%s\" or equivalent\n",
		  i, buf, tests[i].str[0]);
	  ret = 1;
	}
    }

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
