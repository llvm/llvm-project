#include <errno.h>
#include <stdio.h>
#include <unistd.h>

static int
do_test (void)
{
  char *argv[] = { (char *) "does-not-exist", NULL };
  char *envp[] = { (char *) "FOO=BAR", NULL };
  errno = 0;
  execve (argv[0], argv, envp);

  if (errno != ENOENT)
    {
      printf ("errno = %d (%m), expected ENOENT\n", errno);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
