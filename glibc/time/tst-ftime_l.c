#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>


static int
do_test (void)
{
  locale_t l;
  locale_t old;
  struct tm tm;
  char buf[1000];
  wchar_t wbuf[1000];
  int result = 0;
  size_t n;

  l = newlocale (LC_ALL_MASK, "de_DE.ISO-8859-1", NULL);
  if (l == NULL)
    {
      puts ("newlocale failed");
      exit (1);
    }

  memset (&tm, '\0', sizeof (tm));

  tm.tm_year = 102;
  tm.tm_mon = 2;
  tm.tm_mday = 1;

  if (strftime (buf, sizeof (buf), "%e %^B %Y", &tm) == 0)
    {
      puts ("initial strftime failed");
      exit (1);
    }
  if (strcmp (buf, " 1 MARCH 2002") != 0)
    {
      printf ("initial strftime: expected \"%s\", got \"%s\"\n",
	      " 1 MARCH 2002", buf);
      result = 1;
    }
  else
    printf ("got \"%s\"\n", buf);

  /* Now using the extended locale model.  */
  if (strftime_l (buf, sizeof (buf), "%e %^B %Y", &tm, l) == 0)
    {
      puts ("strftime_l failed");
      result = 1;
    }
  else if (strcmp (buf, " 1 M\xc4RZ 2002") != 0)
    {
      printf ("strftime_l: expected \"%s\", got \"%s\"\n",
	      " 1 M\xc4RZ 2002", buf);
      result = 1;
    }
  else
    {
      setlocale (LC_ALL, "de_DE.ISO-8859-1");
      printf ("got \"%s\"\n", buf);
      setlocale (LC_ALL, "C");
    }

  /* And the wide character version.  */
  if (wcsftime_l (wbuf, sizeof (wbuf) / sizeof (wbuf[0]), L"%e %^B %Y", &tm, l)
      == 0)
    {
      puts ("wcsftime_l failed");
      result = 1;
    }
  else if (wcscmp (wbuf, L" 1 M\x00c4RZ 2002") != 0)
    {
      printf ("wcsftime_l: expected \"%ls\", got \"%ls\"\n",
	      L" 1 M\x00c4RZ 2002", wbuf);
      result = 1;
    }
  else
    {
      setlocale (LC_ALL, "de_DE.ISO-8859-1");
      printf ("got \"%ls\"\n", wbuf);
      setlocale (LC_ALL, "C");
    }

  old = uselocale (l);

  n = strftime (buf, sizeof (buf), "%e %^B %Y", &tm);

  /* Switch back.  */
  (void) uselocale (old);

  if (n == 0)
    {
      puts ("strftime after first uselocale failed");
      result = 1;
    }
  else if (strcmp (buf, " 1 M\xc4RZ 2002") != 0)
    {
      printf ("strftime in non-C locale: expected \"%s\", got \"%s\"\n",
	      " 1 M\xc4RZ 2002", buf);
      result = 1;
    }
  else
    {
      setlocale (LC_ALL, "de_DE.ISO-8859-1");
      printf ("got \"%s\"\n", buf);
      setlocale (LC_ALL, "C");
    }

  if (strftime (buf, sizeof (buf), "%e %^B %Y", &tm) == 0)
    {
      puts ("strftime after second uselocale failed");
      result = 1;
    }
  else if (strcmp (buf, " 1 MARCH 2002") != 0)
    {
      printf ("initial strftime: expected \"%s\", got \"%s\"\n",
	      " 1 MARCH 2002", buf);
      result = 1;
    }
  else
    printf ("got \"%s\"\n", buf);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
