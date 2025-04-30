#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
do_test (void)
{
#ifndef __clang__ /* clang never finishes */
  size_t instances = 16384;
#define X0 "\n%1$s\n" "%1$s" "%2$s" "%2$s" "%3$s" "%4$s" "%5$d" "%5$d"
  const char *item = "\na\nabbcd55";
#define X3 X0 X0 X0 X0 X0 X0 X0 X0
#define X6 X3 X3 X3 X3 X3 X3 X3 X3
#define X9 X6 X6 X6 X6 X6 X6 X6 X6
#define X12 X9 X9 X9 X9 X9 X9 X9 X9
#define X14 X12 X12 X12 X12
#define TRAILER "%%%%%%%%%%%%%%%%%%%%%%%%%%"
#define TRAILER2 TRAILER TRAILER
  size_t length = instances * strlen (item) + strlen (TRAILER) + 1;

  char *buf = malloc (length + 1);
  snprintf (buf, length + 1,
	    X14 TRAILER2 "\n",
	    "a", "b", "c", "d", 5);

  const char *p = buf;
  size_t i;
  for (i = 0; i < instances; ++i)
    {
      const char *expected;
      for (expected = item; *expected; ++expected)
	{
	  if (*p != *expected)
	    {
	      printf ("mismatch at offset %zu (%zu): expected %d, got %d\n",
		      (size_t) (p - buf), i, *expected & 0xFF, *p & 0xFF);
	      return 1;
	    }
	  ++p;
	}
    }
  if (strcmp (p, TRAILER "\n") != 0)
    {
      printf ("mismatch at trailer: [%s]\n", p);
      return 1;
    }
  free (buf);
#endif /* __clang__ */
  return 0;
}
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
