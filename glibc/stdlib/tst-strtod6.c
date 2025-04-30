#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tst-strtod.h"

#define TEST_STRTOD(FSUF, FTYPE, FTOSTR, LSUF, CSUF)	  \
static int						  \
test_strto ## FSUF (const char str[])			  \
{							  \
  char *endp;						  \
  int result = 0;					  \
  puts (str);						  \
  FTYPE d = strto ## FSUF (str, &endp);			  \
  if (!isnan (d))					  \
    {							  \
      puts ("strto" #FSUF " did not return NAN");	  \
      result = 1;					  \
    }							  \
  if (issignaling (d))					  \
    {							  \
      puts ("strto" #FSUF " returned a sNAN");		  \
      result = 1;					  \
    }							  \
  if (strcmp (endp, "something") != 0)			  \
    {							  \
      puts ("strto" #FSUF " set incorrect end pointer");  \
      result = 1;					  \
    }							  \
  return result;					  \
}

GEN_TEST_STRTOD_FOREACH (TEST_STRTOD);

static int
do_test (void)
{
  int result = 0;

  result |= STRTOD_TEST_FOREACH (test_strto, "NaN(blabla)something");
  result |= STRTOD_TEST_FOREACH (test_strto, "NaN(1234)something");
  /* UINT32_MAX.  */
  result |= STRTOD_TEST_FOREACH (test_strto, "NaN(4294967295)something");
  /* UINT64_MAX.  */
  result |= STRTOD_TEST_FOREACH (test_strto,
				 "NaN(18446744073709551615)something");
  /* The case of zero is special in that "something" has to be done to make the
     mantissa different from zero, which would mean infinity instead of
     NaN.  */
  result |= STRTOD_TEST_FOREACH (test_strto, "NaN(0)something");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
