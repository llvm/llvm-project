#include <locale.h>
#include <stdio.h>

static int
do_test (void)
{
  locale_t d = duplocale (LC_GLOBAL_LOCALE);
  if (d != (locale_t) 0)
    freelocale (d);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
