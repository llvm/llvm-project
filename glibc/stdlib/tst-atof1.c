#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
do_test (void)
{
  char buf[100];
  snprintf (buf, sizeof (buf), "%g", atof ("0x10p-1"));
  if (strcmp (buf, "8") != 0)
    {
      printf ("got \"%s\", expected \"8\"\n", buf);
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
