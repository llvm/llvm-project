/* BZ #5225 */
#include <stdio.h>
#include <string.h>
#include <wchar.h>

static int
do_test (void)
{
  wchar_t in[] = L"123,abc,321";
  /* This is the critical part for this test.  format must be in
     read-only memory.  */
  static const wchar_t format[50] = L"%d,%[^,],%d";
  int out_d1, out_d2;
  char out_s[50];
  printf ("in='%ls' format='%ls'\n", in, format);
  if (swscanf (in, format, &out_d1, out_s, &out_d2) != 3)
    {
      puts ("swscanf did not return 3");
      return 1;
    }
  printf ("in='%ls' format='%ls'\n", in, format);
  printf ("out_d1=%d out_s='%s' out_d2=%d\n", out_d1, out_s, out_d2);
  if (out_d1 != 123 || strcmp (out_s, "abc") != 0 || out_d2 != 321)
    {
      puts ("swscanf did not return the correct values");
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
