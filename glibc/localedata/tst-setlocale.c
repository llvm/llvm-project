/* Test case by Jakub Jelinek <jakub@redhat.com>.  */
#include <locale.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  char q[30];
  char *s;

  setlocale (LC_ALL, "");
  printf ("after setlocale (LC_ALL, \"\"): %s\n", setlocale(LC_NUMERIC, NULL));

  strcpy (q, "de_DE.UTF-8");
  setlocale (LC_NUMERIC, q);
  printf ("after setlocale (LC_NUMERIC, \"%s\"): %s\n",
	  q, setlocale(LC_NUMERIC, NULL));

  strcpy (q, "de_DE.ISO-8859-1");
  s = setlocale (LC_NUMERIC, NULL);
  printf ("after overwriting string: %s\n", s);

  return strcmp (s, "de_DE.UTF-8") != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
