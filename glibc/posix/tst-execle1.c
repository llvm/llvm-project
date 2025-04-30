#include <errno.h>
#include <stdio.h>
#include <unistd.h>

static int
do_test (void)
{
  static const char prog[] = "does-not-exist";
  const char *env [] = {"FOO=BAR", NULL};
  errno = 0;
  execle (prog, prog, NULL, env);

  if (errno != ENOENT)
    {
      printf ("errno = %d (%m), expected ENOENT\n", errno);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
