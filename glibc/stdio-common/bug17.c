#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  static const char expect[] = "0, 0, 0";
  char buf[100];
  int status = 0;

  static const char fmt1[] = "%0d, %0ld, %0lld";
  snprintf (buf, sizeof (buf), fmt1, 0, 0L, 0LL);
  if (strcmp (buf, expect) != 0)
    {
      printf ("\"%s\": got \"%s\", expected \"%s\"\n", fmt1, buf, expect);
      status = 1;
    }

  static const char fmt2[] = "%0u, %0lu, %0llu";
  snprintf (buf, sizeof (buf), fmt2, 0u, 0uL, 0uLL);
  if (strcmp (buf, expect) != 0)
    {
      printf ("\"%s\": got \"%s\", expected \"%s\"\n", fmt2, buf, expect);
      status = 1;
    }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
