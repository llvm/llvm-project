#include <stdio.h>

static int
do_test (void)
{
  char buf[100];
  int result = 0;

  if (ferror (stdin) != 0)
    {
      fputs ("error bit set for stdin at startup\n", stdout);
      result = 1;
    }
  if (fgets (buf, sizeof buf, stdin) != buf)
    {
      fputs ("fgets with existing input has problem\n", stdout);
      result = 1;
    }
  if (ferror (stdin) != 0)
    {
      fputs ("error bit set for stdin after setup\n", stdout);
      result = 1;
    }
  if (fputc ('a', stdin) != EOF)
    {
      fputs ("fputc to stdin does not terminate with an error\n", stdout);
      result = 1;
    }
  if (ferror (stdin) == 0)
    {
      fputs ("error bit not set for stdin after fputc\n", stdout);
      result = 1;
    }
  clearerr (stdin);
  if (ferror (stdin) != 0)
    {
      fputs ("error bit set for stdin after clearerr\n", stdout);
      result = 1;
    }
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
