#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static const char expected[] = "\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55\
\n\
a\n\
abbcd55%%%%%%%%%%%%%%%%%%%%%%%%%%\n";

static int
do_test (void)
{
  char *buf = malloc (strlen (expected) + 1);
  snprintf (buf, strlen (expected) + 1,
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
	    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",
	    "a", "b", "c", "d", 5);
  return strcmp (buf, expected) != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
