#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <iconv.h>
#include <locale.h>

static const char testbuf[] = {
	0xEF, 0xBE, 0x9F, 0xD0, 0xB4, 0xEF, 0xBE, 0x9F, 0x29, 0xEF, 0xBE, 0x8E,
	0xEF, 0xBE, 0x9F, 0xEF, 0xBD, 0xB6, 0xEF, 0xBD, 0xB0, 0xEF, 0xBE, 0x9D
};

static int
do_test (void)
{
  setlocale (LC_ALL, "de_DE.UTF-8");
  iconv_t ic = iconv_open ("ISO-2022-JP//TRANSLIT", "UTF-8");
  if (ic == (iconv_t) -1)
    {
      puts ("iconv_open failed");
      return 1;
    }
  size_t outremain = sizeof testbuf;
  char outbuf[outremain];
  char *inp = (char *) testbuf;
  char *outp = outbuf;
  size_t inremain = sizeof testbuf;

  int ret = iconv (ic, &inp, &inremain, &outp, &outremain);

  int result = 0;
  if (ret == (size_t) -1)
    {
      if (errno == E2BIG)
	puts ("buffer too small reported.  OK");
      else
	{
	  printf ("iconv failed with %d (%m)\n", errno);
	  result = 0;
	}
    }
  else
    {
      printf ("iconv returned %d\n", ret);
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
