#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static struct
{
  const char *str1;
  const char *str2;
} tests[] =
  {
    { "B0075022800016.gbp.corp.com", "B007502280067.gbp.corp.com" },
    { "B0075022800016.gbp.corp.com", "B007502357019.GBP.CORP.COM" },
    { "B007502280067.gbp.corp.com", "B007502357019.GBP.CORP.COM" }
  };
#define ntests (sizeof (tests) / sizeof (tests[0]))


int
compare (const char *str1, const char *str2, int exp)
{
  int c = strverscmp (str1, str2);
  if (c != 0)
    c /= abs (c);
  return c != exp;
}


int
do_test (void)
{
  int res = 0;
  for (int i = 0; i < ntests; ++i)
    {
      if (compare (tests[i].str1, tests[i].str2, -1))
	{
	  printf ("FAIL: \"%s\" > \"%s\"\n", tests[i].str1, tests[i].str2);
	  res = 1;
	}
      if (compare (tests[i].str2, tests[i].str1, +1))
	{
	  printf ("FAIL: \"%s\" > \"%s\"\n", tests[i].str2, tests[i].str1);
	  res = 1;
	}
      char *copy1 = strdupa (tests[i].str1);
      if (compare (tests[i].str1, copy1, 0))
	{
	  printf ("FAIL: \"%s\" != \"%s\"\n", tests[i].str1, copy1);
	  res = 1;
	}
      char *copy2 = strdupa (tests[i].str2);
      if (compare (tests[i].str2, copy2, 0))
	{
	  printf ("FAIL: \"%s\" != \"%s\"\n", tests[i].str2, copy2);
	  res = 1;
	}
    }
  return res;
}

#include <support/test-driver.c>
