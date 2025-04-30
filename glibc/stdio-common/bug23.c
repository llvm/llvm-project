#include <stdio.h>
#include <string.h>

static char buf[32768];
static const char expected[] = "\
\n\
a\n\
abbcd55%%%%%%%%%%%%%%%%%%%%%%%%%%\n";

static int
do_test (void)
{
  snprintf (buf, sizeof (buf),
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",
	    "a", "b", "c", "d", 5);
  return strcmp (buf, expected) != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
