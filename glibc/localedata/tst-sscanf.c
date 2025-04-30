#include <stdio.h>
#include <locale.h>
#include <assert.h>

#define P0 "\xDB\xB0"
#define P1 "\xDB\xB1"
#define P2 "\xDB\xB2"
#define P3 "\xDB\xB3"
#define P4 "\xDB\xB4"
#define P5 "\xDB\xB5"
#define P6 "\xDB\xB6"
#define P7 "\xDB\xB7"
#define P8 "\xDB\xB8"
#define P9 "\xDB\xB9"
#define PD "\xd9\xab"
#define PT "\xd9\xac"

static int
check_sscanf (const char *s, const char *format, const float n)
{
  float f;

  if (sscanf (s, format, &f) != 1)
    {
      printf ("nothing found for \"%s\"\n", s);
      return 1;
    }
  if (f != n)
    {
      printf ("got %f expected %f from \"%s\"\n", f, n, s);
      return 1;
    }
  return 0;
}

static int
do_test (void)
{
  if (setlocale (LC_ALL, "fa_IR.UTF-8") == NULL)
    {
      puts ("cannot set fa_IR locale");
      return 1;
    }

  int r = check_sscanf (P3 PD P1 P4, "%I8f", 3.14);
  r |= check_sscanf (P3 PT P1 P4 P5, "%I'f", 3145);
  r |= check_sscanf (P3 PD P1 P4 P1 P5 P9, "%If", 3.14159);
  r |= check_sscanf ("-" P3 PD P1 P4 P1 P5, "%If", -3.1415);
  r |= check_sscanf ("+" PD P1 P4 P1 P5, "%If", +.1415);
  r |= check_sscanf (P3 PD P1 P4 P1 P5 "e+" P2, "%Ie", 3.1415e+2);

  return r;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
