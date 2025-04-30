#include <stdio.h>
#include <string.h>

int
do_test (void)
{
  char str[] = "this is a test";

  strfry (str);

  return 0;
}

#include <support/test-driver.c>
