// BZ 12814
#include <errno.h>
#include <iconv.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  iconv_t h = iconv_open ("ISO-2022-JP-2", "UTF-8");
  if (h == (iconv_t) -1)
    {
      printf ("cannot load iconv module: %m\n");
      return 1;
    }

  // Euro sign
  static const char inbuf[] = "\xe2\x82\xac";
  char *in = (char *) inbuf;
  size_t inlen = sizeof (inbuf) - 1;

  char outbuf[100];
  char *out = outbuf;
  size_t outlen = sizeof (outbuf);

  int res = 0;
  size_t n = iconv (h, &in, &inlen, &out, &outlen);
  if (n == (size_t) -1)
    {
      printf ("iconv failed with %d: %m\n", errno);
      return 1;
    }
  if (n != 0)
    {
      printf ("iconv returned %zu, expected zero\n", n);
      res = 1;
    }
  if (in != inbuf + sizeof (inbuf) - 1)
    {
      printf ("in advanced by %td, expected %zu\n",
	      in - inbuf, sizeof (inbuf) - 1);
      res = 1;
    }
  static const char expected[] = "\x1b\x2e\x46\x1b\x4e\x24";
  if (out - outbuf != sizeof (expected) - 1
      || memcmp (outbuf, expected, sizeof (expected) - 1) != 0)
    {
      fputs ("generated sequence is: \"", stdout);
      for (size_t i = 0; i < out - outbuf; ++i)
	printf ("\\x%02hhx", outbuf[i]);
      fputs ("\", expected \"", stdout);
      for (size_t i = 0; i < sizeof (expected) - 1; ++i)
	printf ("\\x%02hhx", expected[i]);
      puts ("\"");
      res = 1;
    }

  if (iconv_close (h) != 0)
    {
      puts ("failed closing iconv module");
      res = 1;
    }

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
