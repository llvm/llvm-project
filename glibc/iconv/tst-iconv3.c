/* Contributed by Owen Taylor <otaylor@redhat.com>.  */

#include <iconv.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>

#define BUFSIZE 10000

static int
do_test (void)
{
  char inbuf[BUFSIZE];
  wchar_t outbuf[BUFSIZE];

  iconv_t cd;
  int i;
  char *inptr;
  char *outptr;
  size_t inbytes_left, outbytes_left;
  int count;
  int result = 0;

  for (i=0; i < BUFSIZE; i++)
    inbuf[i] = 'a';

  cd = iconv_open ("UCS-4LE", "UTF-8");

  inbytes_left = BUFSIZE;
  outbytes_left = BUFSIZE * 4;
  inptr = inbuf;
  outptr = (char *) outbuf;

  count = iconv (cd, &inptr, &inbytes_left, &outptr, &outbytes_left);

  if (count < 0)
    {
      if (errno == E2BIG)
	printf ("Received E2BIG\n");
      else
	printf ("Received something else\n");

      printf ("inptr change: %td\n", inptr - inbuf);
      printf ("inlen change: %zd\n", BUFSIZE - inbytes_left);
      printf ("outptr change: %td\n", outptr - (char *) outbuf);
      printf ("outlen change: %zd\n", BUFSIZE * 4 - outbytes_left);
      result = 1;
    }
  else
    printf ("Succeeded\n");

  iconv_close (cd);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
