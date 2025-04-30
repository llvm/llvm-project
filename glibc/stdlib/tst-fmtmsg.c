#include <fmtmsg.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MM_TEST 10

static int
do_test (void)
{
  int result = 0;

  mtrace ();

  char TEST[] = "ABCD";
  if (addseverity (MM_TEST, TEST) != MM_OK)
    {
      puts ("addseverity failed");
      result = 1;
    }
  strcpy (TEST, "TEST");

  if (fmtmsg (MM_PRINT, "GLIBC:tst-fmtmsg", MM_HALT, "halt",
	      "should print message for MM_HALT", "GLIBC:tst-fmtmsg:1")
      != MM_OK)
    result = 1;

  if (fmtmsg (MM_PRINT, "GLIBC:tst-fmtmsg", MM_ERROR, "halt",
	      "should print message for MM_ERROR", "GLIBC:tst-fmtmsg:2")
      != MM_OK)
    result = 1;

  if (fmtmsg (MM_PRINT, "GLIBC:tst-fmtmsg", MM_WARNING, "halt",
	      "should print message for MM_WARNING", "GLIBC:tst-fmtmsg:3")
      != MM_OK)
    result = 1;

  if (fmtmsg (MM_PRINT, "GLIBC:tst-fmtmsg", MM_INFO, "halt",
	      "should print message for MM_INFO", "GLIBC:tst-fmtmsg:4")
      != MM_OK)
    result = 1;

  if (fmtmsg (MM_PRINT, "GLIBC:tst-fmtmsg", MM_NOSEV, "halt",
	      "should print message for MM_NOSEV", "GLIBC:tst-fmtmsg:5")
      != MM_OK)
    result = 1;

  if (fmtmsg (MM_PRINT, "GLIBC:tst-fmtmsg", MM_TEST, "halt",
	      "should print message for MM_TEST", "GLIBC:tst-fmtmsg:6")
      != MM_OK)
    result = 1;

  if (addseverity (MM_TEST, NULL) != MM_OK)
    {
      puts ("second addseverity failed");
      result = 1;
    }

  if (addseverity (MM_TEST, NULL) != MM_NOTOK)
    {
      puts ("third addseverity unexpectedly succeeded");
      result = 1;
    }

  char *p = strdup ("TEST2");
  if (addseverity (MM_TEST, p) != MM_OK)
    {
      puts ("fourth addseverity failed");
      result = 1;
    }
  if (addseverity (MM_TEST, "TEST3") != MM_OK)
    {
      puts ("fifth addseverity failed");
      result = 1;
    }

  free (p);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
