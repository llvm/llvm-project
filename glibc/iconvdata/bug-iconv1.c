/* Test program by Satoru Takabayashi.  */
#include <errno.h>
#include <iconv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main (int argc, char **argv)
{
  const char in[] = "\x41\x42\x43\xa4\xa2\xa4\xa4\xa4\xa6\xa4\xa8\xa4\xaa";
                  /* valid eucJP string */
  const char exp[] = "\x41\x42\x43\x82\xa0\x82\xa2\x82\xa4";
  size_t outbufsize = 10;
                  /* 10 is too small to store full result (intentional) */
  size_t inleft, outleft;
  char *in_p = (char *) in;
  char out[outbufsize];
  char *out_p = out;
  iconv_t cd;
  int i;

  inleft = strlen (in);
  outleft = outbufsize;

  cd = iconv_open ("SJIS", "eucJP");
  if (cd == (iconv_t) -1)
    {
      puts ("iconv_open failed");
      exit (1);
    }

  iconv (cd, &in_p, &inleft, &out_p, &outleft); /* this returns E2BIG */
  for (i = 0; i < outbufsize - outleft; ++i)
    printf (" %02x", (unsigned char) out[i]);
  puts ("");
  iconv_close (cd);

  return outbufsize - outleft != 9 || memcmp (out, exp, 9) != 0;
}
