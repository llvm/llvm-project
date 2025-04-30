#include <regex.h>
#include <stdio.h>

static int
do_test (void)
{
  regex_t r;
  int e = regcomp(&r, "xy\\{4,5,7\\}zabc", 0);
  char buf[100];
  regerror(e, &r, buf, sizeof (buf));
  printf ("e = %d (%s)\n", e, buf);
  int res = e != REG_BADBR;

  e = regcomp(&r, "xy\\{4,5a\\}zabc", 0);
  regerror(e, &r, buf, sizeof (buf));
  printf ("e = %d (%s)\n", e, buf);
  res |= e != REG_BADBR;

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
