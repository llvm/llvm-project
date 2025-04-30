// BZ 12601
#include <stdio.h>
#include <errno.h>
#include <iconv.h>

static int
do_test (void)
{
   iconv_t cd;
   char in[] = "\x83\xd9";
   char out[256];
   char *inbuf;
   size_t inbytesleft;
   char *outbuf;
   size_t outbytesleft;
   size_t ret;

   inbuf = in;
   inbytesleft = sizeof (in) - 1;
   outbuf = out;
   outbytesleft = sizeof (out);

   cd = iconv_open("utf-8", "cp932");
   ret = iconv(cd, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
   iconv_close(cd);

   printf("result: %zd %d %zd %d\n", ret, errno, inbytesleft, inbuf[0]);

   /*
    * result: -1 84 0 0        (84=EILSEQ)
    *
    * Error is returnd but inbuf is consumed.
    *
    * \x83\xd9 is valid shift-jis sequence but no character is assigned
    * to it.
    */

   return (ret != -1 || errno != EILSEQ
	   || inbytesleft != 2 || inbuf[0] != in[0]);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
