/* Based on a test case by Paul Eggert.  */
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char const string[] = "";


static int
test (const char *locale)
{
  size_t bufsize;
  size_t r;
  size_t l;
  char *buf;
  locale_t loc;
  int result = 0;

  if (setlocale (LC_COLLATE, locale) == NULL)
    {
      printf ("cannot set locale \"%s\"\n", locale);
      return 1;
    }
  bufsize = strxfrm (NULL, string, 0) + 1;
  buf = malloc (bufsize);
  if (buf == NULL)
    {
      printf ("cannot allocate %zd bytes\n", bufsize);
      return 1;
    }
  r = strxfrm (buf, string, bufsize);
  l = strlen (buf);
  if (r != l)
    {
       printf ("locale \"%s\": strxfrm returned %zu, strlen returned %zu\n",
	       locale, r, l);
       result = 1;
    }

  loc = newlocale (1 << LC_ALL, locale, NULL);

  r = strxfrm_l (buf, string, bufsize, loc);
  l = strlen (buf);
  if (r != l)
    {
       printf ("locale \"%s\": strxfrm_l returned %zu, strlen returned %zu\n",
	       locale, r, l);
       result = 1;
    }

  freelocale (loc);

  free (buf);

  return result;
}


int
do_test (void)
{
  int result = 0;

  result |= test ("C");
  result |= test ("en_US.ISO-8859-1");
  result |= test ("de_DE.UTF-8");

  return result;
}

#include <support/test-driver.c>
