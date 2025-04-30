#include <errno.h>
#include <iconv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int
do_test (void)
{
  iconv_t cd = iconv_open ("ISO-8859-1", "UNICODE");
  if (cd == (iconv_t) -1)
    {
      printf ("iconv_open failed: %m\n");
      exit (EXIT_FAILURE);
    }

  char instr[] = "a";
  char *inptr = instr;
  size_t inlen = strlen (instr);
  char buf[200];
  char *outptr = buf;
  size_t outlen = sizeof (outptr);

  errno = 0;
  size_t n = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  if (n != (size_t) -1)
    {
      printf ("n (= %zu) != (size_t) -1\n", n);
      exit (EXIT_FAILURE);
    }
  if (errno != EINVAL)
    {
      printf ("errno = %m, not EINVAL\n");
      exit (EXIT_FAILURE);
    }

  iconv_close (cd);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
