/* tst-strptime2 - Test strptime %z timezone offset specifier.  */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <libc-diag.h>

/* Dummy string is used to match strptime's %s specifier.  */

static const char dummy_string[] = "1113472456";

/* buffer_size contains the maximum test string length, including
   trailing NUL.  */

enum
{
  buffer_size = 20,
};

/* Verbose execution, set with --verbose command line option.  */

static bool verbose;


/* mkbuf - Write a test string for strptime with the specified time
   value and number of digits into the supplied buffer, and return
   the expected strptime test result.

   The test string, buf, is written with the following content:
     a dummy string matching strptime "%s" format specifier,
     whitespace matching strptime " " format specifier, and
     timezone string matching strptime "%z" format specifier.

   Note that a valid timezone string is either "Z" or contains the
   following fields:
     Sign field consisting of a '+' or '-' sign,
     Hours field in two decimal digits, and
     optional Minutes field in two decimal digits. Optionally,
     a ':' is used to seperate hours and minutes.

   This function may write test strings with minutes values outside
   the valid range 00-59.  These are invalid strings and useful for
   testing strptime's rejection of invalid strings.

   The ndigits parameter is used to limit the number of timezone
   string digits to be written and may range from 0 to 4.  Note that
   only 2 and 4 digit strings are valid input to strptime; strings
   with 0, 1 or 3 digits are invalid and useful for testing strptime's
   rejection of invalid strings.

   This function returns the behavior expected of strptime resulting
   from parsing the the test string.  For valid strings, the function
   returns the expected tm_gmtoff value.  For invalid strings,
   LONG_MAX is returned.  LONG_MAX indicates the expectation that
   strptime will return NULL; for example, if the number of digits
   are not correct, or minutes part of the time is outside the valid
   range of 00 to 59.  */

static long int
mkbuf (char *buf, bool neg, bool colon, unsigned int hhmm, size_t ndigits)
{
  const int mm_max = 59;
  char sign = neg ? '-' : '+';
  int i;
  unsigned int hh = hhmm / 100;
  unsigned int mm = hhmm % 100;
  long int expect = LONG_MAX;

  i = sprintf (buf, "%s %c", dummy_string, sign);
#if __GNUC_PREREQ (7, 0)
  /* GCC issues a warning when it thinks the snprintf buffer may be too short.
     This test is explicitly using short buffers to force snprintf to truncate
     the output so we ignore the warnings.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (7.0, "-Wformat-truncation");
#endif
  if (colon)
    snprintf (buf + i, ndigits + 2, "%02u:%02u", hh, mm);
  else
    snprintf (buf + i, ndigits + 1, "%04u", hhmm);
#if __GNUC_PREREQ (7, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif

  if (mm <= mm_max && (ndigits == 2 || ndigits == 4))
    {
      long int tm_gmtoff = hh * 3600 + mm * 60;

      expect = neg ? -tm_gmtoff : tm_gmtoff;
    }

  return expect;
}


/* Write a description of expected or actual test result to stdout.  */

static void
describe (bool string_valid, long int tm_gmtoff)
{
  if (string_valid)
    printf ("valid, tm.tm_gmtoff %ld", tm_gmtoff);
  else
    printf ("invalid, return value NULL");
}


/* Using buffer buf, run strptime.  Compare results against expect,
  the expected result.  Report failures and verbose results to stdout.
  Update the result counts.  Return 1 if test failed, 0 if passed.  */

static int
compare (const char *buf, long int expect, unsigned int *nresult)
{
  struct tm tm;
  char *p;
  bool test_string_valid;
  long int test_result;
  bool fail;
  int result;

  p = strptime (buf, "%s %z", &tm);
  test_string_valid = p != NULL;
  test_result = test_string_valid ? tm.tm_gmtoff : LONG_MAX;
  fail = test_result != expect;

  if (fail || verbose)
    {
      bool expect_string_valid = expect != LONG_MAX;

      printf ("%s: input \"%s\", expected: ", fail ? "FAIL" : "PASS", buf);
      describe (expect_string_valid, expect);

      if (fail)
	{
	  printf (", got: ");
	  describe (test_string_valid, test_result);
	}

      printf ("\n");
    }

  result = fail ? 1 : 0;
  nresult[result]++;

  return result;
}


static int
do_test (void)
{
  char buf[buffer_size];
  long int expect;
  int result = 0;
  /* Number of tests run with passing (index==0) and failing (index==1)
     results.  */
  unsigned int nresult[2];
  unsigned int ndigits;
  unsigned int step;
  unsigned int hhmm;

  nresult[0] = 0;
  nresult[1] = 0;

  /* Create and test input string with no sign and four digits input
     (invalid format).  */

  sprintf (buf, "%s  1030", dummy_string);
  expect = LONG_MAX;
  result |= compare (buf, expect, nresult);

  /* Create and test input string with "Z" input (valid format).
     Expect tm_gmtoff of 0.  */

  sprintf (buf, "%s Z", dummy_string);
  expect = 0;
  result |= compare (buf, expect, nresult);

  /* Create and test input strings with sign and digits:
     0 digits (invalid format),
     1 digit (invalid format),
     2 digits (valid format),
     3 digits (invalid format),
     4 digits (valid format if and only if minutes is in range 00-59,
	       otherwise invalid).
     If format is valid, the returned tm_gmtoff is checked.  */

  for (ndigits = 0, step = 10000; ndigits <= 4; ndigits++, step /= 10)
    for (hhmm = 0; hhmm <= 9999; hhmm += step)
      {
	/* Test both positive and negative signs.  */

	expect = mkbuf (buf, false, false, hhmm, ndigits);
	result |= compare (buf, expect, nresult);

	expect = mkbuf (buf, true, false, hhmm, ndigits);
	result |= compare (buf, expect, nresult);

	/* Test with colon as well.  */

	if (ndigits >= 3)
	  {
	    expect = mkbuf (buf, false, true, hhmm, ndigits);
	    result |= compare (buf, expect, nresult);

	    expect = mkbuf (buf, true, true, hhmm, ndigits);
	    result |= compare (buf, expect, nresult);
	  }
      }

  if (result > 0 || verbose)
    printf ("%s: %u input strings: %u fail, %u pass\n",
	    result > 0 ? "FAIL" : "PASS",
	    nresult[1] + nresult[0], nresult[1], nresult[0]);

  return result;
}


/* Add a "--verbose" command line option to test-skeleton.c.  */

#define OPT_VERBOSE 10000

#define CMDLINE_OPTIONS \
  { "verbose", no_argument, NULL, OPT_VERBOSE, },

#define CMDLINE_PROCESS \
  case OPT_VERBOSE: \
    verbose = true; \
    break;

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
