/* Test case for setlocale vs uselocale (LC_GLOBAL_LOCALE) bug.  */

#define _GNU_SOURCE 1
#include <locale.h>
#include <stdio.h>
#include <ctype.h>

static int
do_test (void)
{
  locale_t loc_new, loc_old;

  int first = !!isalpha(0xE4);

  setlocale (LC_ALL, "de_DE");

  int global_de = !!isalpha(0xE4);

  loc_new = newlocale (1 << LC_ALL, "C", 0);
  loc_old = uselocale (loc_new);

  int used_c = !!isalpha(0xE4);

  uselocale (loc_old);

  int used_global = !!isalpha(0xE4);

  printf ("started %d, after setlocale %d\n", first, global_de);
  printf ("after uselocale %d, after LC_GLOBAL_LOCALE %d\n",
	  used_c, used_global);

  freelocale (loc_new);
  return !(used_c == first && used_global == global_de);
}


#define TEST_FUNCTION do_test ()
#include "test-skeleton.c"
