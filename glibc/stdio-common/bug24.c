#include <stdio.h>
#include <errno.h>

static int
do_test (void)
{
  FILE *fp = fopen ("/foobar_does_no_exit", "re");
  if (fp != NULL)
    {
      /* A joker created this file.  Ignore the test.  */
      fclose (fp);
      return 0;
    }

  if (errno == ENOENT)
    {
      printf ("no bug\n");
      return 0;
    }

  printf ("bug : expected ENOENT, got: %m\n");
  return 1;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
