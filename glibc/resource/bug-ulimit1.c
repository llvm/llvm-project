#include <ulimit.h>
#include <stdio.h>

int
main (void)
{
  int retval = 0;
  long int res;

  res = ulimit (UL_SETFSIZE, 10000);
  printf ("Result of ulimit (UL_SETFSIZE, 10000): %ld\n", res);
  if (res != 10000)
    retval = 1;

  res = ulimit (UL_GETFSIZE);
  printf ("Result of ulimit(UL_GETFSIZE): %ld\n", res);
  if (res != 10000)
    retval = 1;

  return retval;
}
