#include <locale.h>
#include <nl_types.h>
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  nl_catd catalog;
  setlocale (LC_ALL, "");

  printf ("LC_MESSAGES = %s\n", setlocale (LC_MESSAGES, NULL));

  catalog = catopen ("sample", NL_CAT_LOCALE);
  if (catalog == (nl_catd) -1)
    {
      printf ("no catalog: %m\n");
      exit (1);
    }

  printf ("%s\n", catgets(catalog, 1, 1, "sample 1"));
  printf ("%s\n", catgets(catalog, 1, 2, "sample 2"));
  printf ("%s\n", catgets(catalog, 1, 3, "sample 3"));
  printf ("%s\n", catgets(catalog, 1, 4, "sample 4"));
  printf ("%s\n", catgets(catalog, 1, 5, "sample 5"));
  printf ("%s\n", catgets(catalog, 1, 6, "sample 6"));
  printf ("%s\n", catgets(catalog, 1, 7, "sample 7"));
  catclose (catalog);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
