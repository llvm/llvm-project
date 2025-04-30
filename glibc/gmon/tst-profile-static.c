#include <stdio.h>
static int
do_test (void)
{
  puts ("Hello world");
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
