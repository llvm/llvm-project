#define _GNU_SOURCE 1
#include <wchar.h>
#include <stdio.h>
#include <string.h>
#include <wctype.h>

static int
do_test (void)
{
  wchar_t tmp[3];
  tmp[0] = '8';
  tmp[1] = '1';
  tmp[2] = 0;

  printf ("Test for wide character output with printf\n");

  printf ("with %%S: %S\n", tmp);

  printf ("with %%C: %C\n", (wint_t) tmp[0]);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
