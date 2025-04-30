// Derived from BZ #9793
#include <errno.h>
#include <iconv.h>
#include <stdio.h>


static int
do_test (void)
{
  iconv_t cd = iconv_open ("ASCII//TRANSLIT", "UTF-8");
  if (cd == (iconv_t) -1)
    {
      puts ("iconv_open failed");
      return 1;
    }

  char input[2] = { 0xc2, 0xae };	/* Registered trademark */
  char *inptr = input;
  size_t insize = sizeof (input);
  char output[2];			/* Too short to contain "(R)".  */
  char *outptr = output;
  size_t outsize = sizeof (output);

  size_t ret = iconv (cd, &inptr, &insize, &outptr, &outsize);
  if (ret != (size_t) -1)
    {
      puts ("iconv succeeded");
      return 1;
    }
  if (errno != E2BIG)
    {
      puts ("iconv did not set errno to E2BIG");
      return 1;
    }
  int res = 0;
  if (inptr != input)
    {
      puts ("inptr changed");
      res = 1;
    }
  if (insize != sizeof (input))
    {
      puts ("insize changed");
      res = 1;
    }
  if (outptr != output)
    {
      puts ("outptr changed");
      res = 1;
    }
  if (outsize != sizeof (output))
    {
      puts ("outsize changed");
      res = 1;
    }
  if (iconv_close (cd) == -1)
    {
      puts ("iconv_close failed");
      res = 1;
    }
  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
