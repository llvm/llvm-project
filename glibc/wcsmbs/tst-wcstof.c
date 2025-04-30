#define _GNU_SOURCE 1
#include <wchar.h>
#include <stdio.h>
#include <string.h>
#include <wctype.h>
#include <libc-diag.h>

static int
do_test (void)
{
  int result = 0;
  char buf[100];
  wchar_t tmp[3];
  tmp[0] = '8';
  tmp[1] = '1';
  tmp[2] = 0;

  /* GCC does not know the result of wcstof so cannot see that the
     snprintf output is not truncated.  */
  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  DIAG_IGNORE_NEEDS_COMMENT (7.0, "-Wformat-truncation");
#endif
  snprintf (buf, 100, "%S = %f", tmp, wcstof (tmp, NULL));
  DIAG_POP_NEEDS_COMMENT;
  printf ("\"%s\" -> %s\n", buf,
	  strcmp (buf, "81 = 81.000000") == 0 ? "okay" : "buggy");
  result |= strcmp (buf, "81 = 81.000000") != 0;

  return result;
}

#include <support/test-driver.c>
