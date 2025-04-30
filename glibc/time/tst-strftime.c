#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static int
do_bz18985 (void)
{
  char buf[1000];
  struct tm ttm;
  int rc, ret = 0;

  memset (&ttm, 1, sizeof (ttm));
  ttm.tm_zone = NULL;  /* Dereferenced directly if non-NULL.  */
  rc = strftime (buf, sizeof (buf), "%a %A %b %B %c %z %Z", &ttm);

  if (rc == 66)
    {
      const char expected[]
	= "? ? ? ? ? ? 16843009 16843009:16843009:16843009 16844909 +467836 ?";
      if (0 != strcmp (buf, expected))
	{
	  printf ("expected:\n  %s\ngot:\n  %s\n", expected, buf);
	  ret += 1;
	}
    }
  else
    {
      printf ("expected 66, got %d\n", rc);
      ret += 1;
    }

  /* Check negative values as well.  */
  memset (&ttm, 0xFF, sizeof (ttm));
  ttm.tm_zone = NULL;  /* Dereferenced directly if non-NULL.  */
  rc = strftime (buf, sizeof (buf), "%a %A %b %B %c %z %Z", &ttm);

  if (rc == 30)
    {
      const char expected[] = "? ? ? ? ? ? -1 -1:-1:-1 1899  ";
      if (0 != strcmp (buf, expected))
	{
	  printf ("expected:\n  %s\ngot:\n  %s\n", expected, buf);
	  ret += 1;
	}
    }
  else
    {
      printf ("expected 30, got %d\n", rc);
      ret += 1;
    }

  return ret;
}

static struct
{
  const char *fmt;
  size_t min;
  size_t max;
} tests[] =
  {
    { "%2000Y", 2000, 4000 },
    { "%02000Y", 2000, 4000 },
    { "%_2000Y", 2000, 4000 },
    { "%-2000Y", 2000, 4000 },
  };
#define ntests (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  size_t cnt;
  int result = 0;

  time_t tnow = time (NULL);
  struct tm *now = localtime (&tnow);

  for (cnt = 0; cnt < ntests; ++cnt)
    {
      size_t size = 0;
      int res;
      char *buf = NULL;

      do
	{
	  size += 500;
	  buf = (char *) realloc (buf, size);
	  if (buf == NULL)
	    {
	      puts ("out of memory");
	      exit (1);
	    }

	  res = strftime (buf, size, tests[cnt].fmt, now);
	  if (res != 0)
	    break;
	}
      while (size < tests[cnt].max);

      if (res == 0)
	{
	  printf ("%Zu: %s: res == 0 despite size == %Zu\n",
		  cnt, tests[cnt].fmt, size);
	  result = 1;
	}
      else if (size < tests[cnt].min)
	{
	  printf ("%Zu: %s: size == %Zu was enough\n",
		  cnt, tests[cnt].fmt, size);
	  result = 1;
	}
      else
	printf ("%Zu: %s: size == %Zu: OK\n", cnt, tests[cnt].fmt, size);

      free (buf);
    }

  struct tm ttm =
    {
      /* Initialize the fields which are needed in the tests.  */
      .tm_mday = 1,
      .tm_hour = 2
    };
  const struct
  {
    const char *fmt;
    const char *exp;
    size_t n;
  } ftests[] =
    {
      { "%-e", "1", 1 },
      { "%-k", "2", 1 },
      { "%-l", "2", 1 },
    };
#define nftests (sizeof (ftests) / sizeof (ftests[0]))
  for (cnt = 0; cnt < nftests; ++cnt)
    {
      char buf[100];
      size_t r = strftime (buf, sizeof (buf), ftests[cnt].fmt, &ttm);
      if (r != ftests[cnt].n)
	{
	  printf ("strftime(\"%s\") returned %zu not %zu\n",
		  ftests[cnt].fmt, r, ftests[cnt].n);
	  result = 1;
	}
      if (strcmp (buf, ftests[cnt].exp) != 0)
	{
	  printf ("strftime(\"%s\") produced \"%s\" not \"%s\"\n",
		  ftests[cnt].fmt, buf, ftests[cnt].exp);
	  result = 1;
	}
    }

  return result + do_bz18985 ();
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
