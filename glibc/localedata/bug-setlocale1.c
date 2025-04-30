// BZ 12788
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static int
do_test (void)
{
  int result = 0;

  char *a = setlocale (LC_ALL, "");
  printf ("setlocale(LC_ALL, \"\") = %s\n", a);
  if (a == NULL)
    return 1;
  a = strdupa (a);

  char *b = setlocale (LC_CTYPE, "");
  printf ("setlocale(LC_CTYPE, \"\") = %s\n", b);
  if (b == NULL)
    return 1;

  char *c = setlocale (LC_ALL, NULL);
  printf ("setlocale(LC_ALL, NULL) = %s\n", c);
  if (c == NULL)
    return 1;
  c = strdupa (c);

  if (strcmp (a, c) != 0)
    {
      puts ("*** first and third result do not match");
      result = 1;
    }

  char *d = setlocale (LC_NUMERIC, "");
  printf ("setlocale(LC_NUMERIC, \"\") = %s\n", d);
  if (d == NULL)
    return 1;

  if (strcmp (d, "C") != 0)
    {
      puts ("*** LC_NUMERIC not C");
      result = 1;
    }

  char *e = setlocale (LC_ALL, NULL);
  printf ("setlocale(LC_ALL, NULL) = %s\n", e);
  if (e == NULL)
    return 1;

  if (strcmp (a, e) != 0)
    {
      puts ("*** first and fifth result do not match");
      result = 1;
    }

  char *f = setlocale (LC_ALL, "C");
  printf ("setlocale(LC_ALL, \"C\") = %s\n", f);
  if (f == NULL)
    return 1;

  if (strcmp (f, "C") != 0)
    {
      puts ("*** LC_ALL not C");
      result = 1;
    }

  char *g = setlocale (LC_ALL, NULL);
  printf ("setlocale(LC_ALL, NULL) = %s\n", g);
  if (g == NULL)
    return 1;

  if (strcmp (g, "C") != 0)
    {
      puts ("*** LC_ALL not C");
      result = 1;
    }

  char *h = setlocale (LC_CTYPE, NULL);
  printf ("setlocale(LC_CTYPE, NULL) = %s\n", h);
  if (h == NULL)
    return 1;

  if (strcmp (h, "C") != 0)
    {
      puts ("*** LC_CTYPE not C");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
