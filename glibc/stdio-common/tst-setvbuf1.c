#include <stdio.h>

static int
do_test (void)
{
  if (setvbuf (stderr, NULL, _IOFBF, BUFSIZ) != 0)
    {
      puts ("Set full buffer error.");
      return 1;
    }

  fprintf (stderr, "Output #1 <stderr>.\n");
  printf ("Output #2 <stdout>.\n");

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
