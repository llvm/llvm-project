#include <langinfo.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  const char *s = nl_langinfo (_NL_LOCALE_NAME (LC_CTYPE));
  if (s == NULL || strcmp (s, "C") != 0)
    {
      printf ("incorrect locale name returned: %s, expected \"C\"\n", s);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
