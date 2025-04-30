#include <errno.h>
#include <stdio.h>
#include <unistd.h>

static int
do_test (void)
{
  static const char prog[] = "does-not-exist";
  errno = 0;
  execl (prog, prog, NULL);

  if (errno != ENOENT)
    {
      printf ("errno = %d (%m), expected ENOENT\n", errno);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
