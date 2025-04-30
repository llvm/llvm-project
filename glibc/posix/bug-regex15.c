/* Test for memory/CPU leak in regcomp.  */

#include <error.h>
#include <sys/types.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_DATA_LIMIT (32 << 20)

static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static int
do_test (void)
{
  regex_t re;
  int reerr;

  reerr = regcomp (&re, "^6?3?[25]?5?[14]*[25]*[69]*+[58]*87?4?$",
		   REG_EXTENDED | REG_NOSUB);
  if (reerr != 0)
    {
      char buf[100];
      regerror (reerr, &re, buf, sizeof buf);
      printf ("regerror %s\n", buf);
      return 1;
    }

  return 0;
}
