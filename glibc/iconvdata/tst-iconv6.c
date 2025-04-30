/* BZ #2569 */

#include <iconv.h>
#include <stdio.h>

static int
do_test (void)
{
  iconv_t cd0 = iconv_open ("ISO-8859-7", "UTF-16LE");
  if (cd0 == (iconv_t) -1)
    {
      puts ("first iconv_open failed");
      return 1;
    }
  iconv_t cd1 = iconv_open ("ISO-8859-7", "UTF-16LE");
  if (cd1 == (iconv_t) -1)
    {
      puts ("second iconv_open failed");
      return 1;
    }
  if (iconv_close (cd0) != 0)
    {
      puts ("first iconv_close failed");
      return 1;
    }
  if (iconv_close (cd1) != 0)
    {
      puts ("second iconv_close failed");
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
