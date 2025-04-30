#include <iconv.h>
#include <stdint.h>
#include <stdio.h>


static int
do_test (void)
{
  iconv_t cd = iconv_open ("utf-8", "unicode");
  if (cd == (iconv_t) -1)
    {
      puts ("cannot open iconv module");
      return 1;
    }

  static const uint16_t us[] = { 0xfeff, 0x0041, 0x0042, 0x0043 };
  char buf[100];

  char *inbuf;
  size_t inlen;
  char *outbuf;
  size_t outlen;
  size_t n;

  inbuf = (char *) us;
  inlen = sizeof (us);
  outbuf = buf;
  outlen = sizeof (buf);
  n = iconv (cd, &inbuf, &inlen, &outbuf, &outlen);
  if (n == (size_t) -1 || inlen != 0 || outlen != sizeof (buf) - 3)
    {
      puts ("first conversion failed");
      return 1;
    }

  iconv (cd, NULL, NULL, NULL, NULL);

  inbuf = (char *) us;
  inlen = sizeof (us);
  outbuf = buf;
  outlen = sizeof (buf);
  n = iconv (cd, &inbuf, &inlen, &outbuf, &outlen);
  if (n == (size_t) -1 || inlen != 0 || outlen != sizeof (buf) - 3)
    {
      puts ("second conversion failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
