/* Dereived from the test case in BZ #2337.  */
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>


static char buf[512] __attribute__ ((aligned (4096)));


static int
do_test (void)
{
  setlocale (LC_ALL, "de_DE.UTF-8");

  FILE *fp = fdopen (dup (STDOUT_FILENO), "a");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "fdopen(,\"a\")");

  setvbuf (fp, buf, _IOFBF, sizeof (buf));

  /* fwprintf to unbuffered stream.   */
  fwprintf (fp, L"hello.\n");

  fclose (fp);

  /* touch my buffer */
  buf[45] = 'a';

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
