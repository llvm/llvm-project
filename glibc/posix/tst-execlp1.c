#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int
do_test (void)
{
  char *cwd = get_current_dir_name ();
  if (cwd == NULL)
    {
      puts ("get_current_dir_name failed");
      return 1;
    }

  /* Make sure we do not find a binary with the name we are going to
     use.  */
  setenv ("PATH", cwd, 1);

  static const char prog[] = "does-not-exist";
  errno = 0;
  execlp (prog, prog, NULL);

  if (errno != ENOENT)
    {
      printf ("errno = %d (%m), expected ENOENT\n", errno);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
