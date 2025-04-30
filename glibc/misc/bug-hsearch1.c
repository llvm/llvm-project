#include <search.h>
#include <stdio.h>

static int
do_test (void)
{
  if (hcreate (1) == 0)
    {
      puts ("hcreate failed");
      return 1;
    }
  ENTRY e;
  e.key = (char *) "a";
  e.data = (char *) "b";
  if (hsearch (e, ENTER) == NULL)
    {
      puts ("ENTER failed");
      return 1;
    }
  ENTRY s;
  s.key = (char *) "c";
  if (hsearch (s, FIND) != NULL)
    {
      puts ("FIND succeeded");
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
