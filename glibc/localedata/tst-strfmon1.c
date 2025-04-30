#include <monetary.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>

static const struct
{
  const char *locale;
  const char *expected;
} tests[] =
  {
    { "de_DE.ISO-8859-1", "|-12,34 EUR|-12,34|" },
    { "da_DK.ISO-8859-1", "|kr. -12,34|-12,34|" },
    { "zh_TW.EUC-TW", "|-NT$12.34|-12.34|" },
    { "sv_SE.ISO-8859-1", "|-12,34 kr|-12,34|" },
    { "nl_NL.UTF-8", "|\u20ac -12,34|-12,34|" },
  };
#define ntests (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  int res = 0;
  for (int i = 0; i < ntests; ++i)
    {
      char buf[500];
      if (setlocale (LC_ALL, tests[i].locale) == NULL)
	{
	  printf ("failed to set locale %s\n", tests[i].locale);
	  res = 1;
	  continue;
	}
      strfmon (buf, sizeof (buf), "|%n|%!n|", -12.34, -12.34);
      int fail = strcmp (buf, tests[i].expected) != 0;
      printf ("%s%s\n", buf, fail ? " *** FAIL ***" : "");
      res |= fail;
    }
  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
