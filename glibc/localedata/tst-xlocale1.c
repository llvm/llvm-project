#include <locale.h>
#include <stdio.h>
#include <string.h>


static struct
{
  const char *locale;
  const char *str1;
  const char *str2;
  int result;
} tests[] =
  {
    { "C", "TRANSLIT", "translit", 0 },
    { "de_DE.ISO-8859-1", "TRANSLIT", "translit", 0 },
    { "de_DE.ISO-8859-1", "TRANSLIT", "trÄnslit", -1 },
    { "de_DE.UTF-8", "TRANSLIT", "translit", 0 },
    { "de_DE.ISO-8859-1", "ä", "Ä", 1 }
  };
#define ntests (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  size_t cnt;
  int result = 0;
  locale_t loc = newlocale (1 << LC_ALL, "C", NULL);

  for (cnt = 0; cnt < ntests; ++cnt)
    {
      int r;

      if (setlocale (LC_ALL, tests[cnt].locale) == NULL)
	{
	  printf ("cannot set locale \"%s\": %m\n", tests[cnt].locale);
	  result = 1;
	  continue;
	}

      printf ("\nstrcasecmp_l (\"%s\", \"%s\", loc)\n",
	      tests[cnt].str1, tests[cnt].str2);

      r = strcasecmp_l (tests[cnt].str1, tests[cnt].str2, loc);
      if (tests[cnt].result == 0)
	{
	  if (r != 0)
	    {
	      printf ("\"%s\" and \"%s\" expected to be the same, result %d\n",
		      tests[cnt].str1, tests[cnt].str2, r);
	      result = 1;
	    }
	}
      else if (tests[cnt].result < 0)
	{
	  if (r >= 0)
	    {
	      printf ("\"%s\" expected to be smaller than \"%s\", result %d\n",
		      tests[cnt].str1, tests[cnt].str2, r);
	      result = 1;
	    }
	}
      else
	{
	  if (r <= 0)
	    {
	      printf ("\"%s\" expected to be larger than \"%s\", result %d\n",
		      tests[cnt].str1, tests[cnt].str2, r);
	      result = 1;
	    }
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
