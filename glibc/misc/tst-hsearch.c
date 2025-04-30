#include <search.h>
#include <stdio.h>

static int
do_test (void)
{
  int a = 1;
  int b = 2;
  ENTRY i;
  ENTRY *e;

  if (hcreate (20) == 0)
    {
      puts ("hcreate failed");
      return 1;
    }

  i.key = (char *) "one";
  i.data = &a;
  if (hsearch (i, ENTER) == NULL)
    return 1;

  i.key = (char *) "one";
  i.data = &b;
  e = hsearch (i, ENTER);
  printf ("e.data = %d\n", *(int *) e->data);
  if (*(int *) e->data != 1)
    return 1;

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
