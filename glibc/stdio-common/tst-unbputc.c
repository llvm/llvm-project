#include <stdio.h>


static int
do_test (void)
{
  putc ('1', stderr);
  putc ('2', stderr);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
