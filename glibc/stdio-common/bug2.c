#include <stdio.h>

int
main (void)
{
  int i;
  puts ("This should print \"wow = I\" for I from 0 to 39 inclusive.");
  for (i = 0; i < 40; i++)
    printf ("%s = %d\n", "wow", i);
  return 0;
}
