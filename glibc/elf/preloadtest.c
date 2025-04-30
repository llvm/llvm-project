#include <stdio.h>

#include "testobj.h"

int
main (void)
{
  int res = preload (42);

  printf ("preload (42) = %d, %s\n", res, res == 92 ? "ok" : "wrong");

  return res != 92;
}

int
foo (int a)
{
  return a;
}
