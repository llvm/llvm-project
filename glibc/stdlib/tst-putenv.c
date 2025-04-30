#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  char *p = getenv ("SOMETHING_NOBODY_USES");
  if (p == NULL)
    {
      puts ("envvar not defined");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
