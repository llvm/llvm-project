/* Test program for %a printf formats.  */

#include <array_length.h>
#include <stdio.h>
#include <string.h>

#ifndef WIDE
# define STR_LEN strlen
# define STR_CMP strcmp
# define SPRINT snprintf
# define CHAR_T char
# define PRINT printf
# define L_(Str) Str
# define S "%s"
#else
# define STR_LEN wcslen
# define SPRINT swprintf
# define STR_CMP wcscmp
# define CHAR_T wchar_t
# define PRINT wprintf
# define L_(Str) L##Str
# define S "%ls"
#endif

struct testcase
{
  double value;
  const CHAR_T *fmt;
  const CHAR_T *expect;
};

static const struct testcase testcases[] =
  {
    { 0x0.0030p+0, L_("%a"),		L_("0x1.8p-11") },
    { 0x0.0040p+0, L_("%a"),		L_("0x1p-10") },
    { 0x0.0030p+0, L_("%040a"),		L_("0x00000000000000000000000000000001.8p-11") },
    { 0x0.0040p+0, L_("%040a"),		L_("0x0000000000000000000000000000000001p-10") },
    { 0x0.0040p+0, L_("%40a"),		L_("                                 0x1p-10") },
    { 0x0.0040p+0, L_("%#40a"),		L_("                                0x1.p-10") },
    { 0x0.0040p+0, L_("%-40a"),		L_("0x1p-10                                 ") },
    { 0x0.0040p+0, L_("%#-40a"),	L_("0x1.p-10                                ") },
    { 0x0.0030p+0, L_("%040e"),		L_("00000000000000000000000000007.324219e-04") },
    { 0x0.0040p+0, L_("%040e"),		L_("00000000000000000000000000009.765625e-04") },
  };


static int
do_test (void)
{
  const struct testcase *t;
  int result = 0;

  for (t = testcases; t < array_end (testcases); ++t)
    {
      CHAR_T buf[1024];
      int n = SPRINT (buf, array_length (buf), t->fmt, t->value);
      if (n != STR_LEN (t->expect) || STR_CMP (buf, t->expect) != 0)
	{
	  PRINT (L_("" S "\tExpected \"" S "\" (%Zu)\n\tGot      \""
		    S "\" (%d, %Zu)\n"),
		 t->fmt, t->expect, STR_LEN (t->expect),
		 buf, n, STR_LEN (buf));
	  result = 1;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
