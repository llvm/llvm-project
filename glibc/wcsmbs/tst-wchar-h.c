#include <stdlib.h>
#include <wchar.h>

static int
do_test (void)
{
  mbstate_t x;
  return sizeof (x) - sizeof (mbstate_t);
}

#include <support/test-driver.c>
