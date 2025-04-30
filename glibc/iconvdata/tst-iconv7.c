#include <iconv.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
do_test (void)
{
  setlocale (LC_ALL, "de_DE.UTF-8");

  iconv_t cd = iconv_open ("ISO-2022-JP//TRANSLIT", "");
  if (cd == (iconv_t) -1)
    {
      puts ("iconv_open failed");
      return 1;
    }

  char instr1[] = "\xc2\xa3\xe2\x82\xac\n";
  const char expstr1[] = "\033$B!r\033(BEUR\n";
  char outstr[32];
  size_t inlen = sizeof (instr1);
  size_t outlen = sizeof (outstr);
  char *inptr = instr1;
  char *outptr = outstr;
  size_t r = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  if (r != 1
      || inlen != 0
      || outlen != sizeof (outstr) - sizeof (expstr1)
      || memcmp (outstr, expstr1, sizeof (expstr1)) != 0)
    {
      puts ("wrong first conversion");
      return 1;
    }

  char instr2[] = "\xe3\x88\xb1\n";
  const char expstr2[] = "(\033$B3t\033(B)\n";
  inlen = sizeof (instr2);
  outlen = sizeof (outstr);
  inptr = instr2;
  outptr = outstr;
  r = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  if (r != 1
      || inlen != 0
      || outlen != sizeof (outstr) - sizeof (expstr2)
      || memcmp (outstr, expstr2, sizeof (expstr2)) != 0)
    {
      puts ("wrong second conversion");
      return 1;
    }

  if (iconv_close (cd) != 0)
    {
      puts ("iconv_close failed");
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
