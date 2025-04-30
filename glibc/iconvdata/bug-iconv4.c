/* Contributed by Jiro SEKIBA <jir@yamato.ibm.com>.  */
#include <errno.h>
#include <iconv.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UCS_STR "\x4e\x8c" /* EUC-TW 0xa2a2, EUC-JP 0x */

static const char *to_code;

static bool
xiconv (iconv_t cd, int out_size)
{
  unsigned char euc[4];
  char *inp = (char *) UCS_STR;
  char *outp = (char *) euc;
  size_t inbytesleft = strlen (UCS_STR);
  size_t outbytesleft = out_size;
  size_t ret;
  bool fail = false;

  errno = 0;
  ret = iconv (cd, &inp, &inbytesleft, &outp, &outbytesleft);
  if (errno || ret == (size_t) -1)
    {
      fail = out_size == 4 || errno != E2BIG;
      printf ("expected %d (E2BIG), got %d (%m)\n", E2BIG, errno);
    }
  else
    {
      printf ("%s: 0x%02x%02x\n", to_code, euc[0], euc[1]);
      if (out_size == 1)
	fail = true;
    }

  return fail;
}


static iconv_t
xiconv_open (const char *code)
{
  iconv_t cd;
  to_code = code;
  errno = 0;
  if (errno || (cd = iconv_open (to_code, "UCS-2BE")) == (iconv_t) -1)
    {
      puts ("Can't open converter");
      exit (1);
    }
  return cd;
}


int
main (void)
{
  iconv_t cd;
  int result = 0;

  cd = xiconv_open ("EUC-TW");
  result |= xiconv (cd, 4) == true;
  puts ("---");
  result |= xiconv (cd, 1) == true;
  puts ("---");
  iconv_close (cd);

  cd = xiconv_open ("EUC-JP");
  result |= xiconv (cd, 4) == true;
  puts ("---");
  result |= xiconv (cd, 1) == true;
  puts ("---");
  iconv_close (cd);

  return result;
}
