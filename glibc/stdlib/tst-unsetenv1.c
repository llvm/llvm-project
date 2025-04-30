#include <stdlib.h>

static int
do_test (void)
{
  clearenv ();
  unsetenv ("FOO");
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
